//! WizGTO solver server — local HTTP wrapper around postflop-solver.
//! API contract: docs/SOLVER_API.md. Listens on 127.0.0.1:3117.

mod cache;
mod solver;

use postflop_solver::{card_from_str, flop_from_str, BoardState, CardConfig, Range, NOT_DEALT};
use serde::Deserialize;
use serde_json::{json, Value};
use solver::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tiny_http::{Header, Method, Response, Server};

const VERSION: &str = "0.11.0";
const ADDR: &str = "127.0.0.1:3117";

fn main() {
    // `wizgto-solver precompute <config.json> [--line N] [--chunk I/K]`
    // fills the flop cache and exits (flags let CI split the work).
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).is_some_and(|a| a == "precompute") {
        let usage = "usage: wizgto-solver precompute <config.json> [--line N] [--chunk I/K]";
        let Some(config) = args.get(2) else {
            eprintln!("{usage}");
            std::process::exit(2);
        };
        let mut line = None;
        let mut chunk = None;
        let mut i = 3;
        while i < args.len() {
            match (args[i].as_str(), args.get(i + 1)) {
                ("--line", Some(v)) => line = v.parse::<usize>().ok(),
                ("--chunk", Some(v)) => {
                    chunk = v
                        .split_once('/')
                        .and_then(|(a, b)| Some((a.parse().ok()?, b.parse().ok()?)));
                }
                _ => {
                    eprintln!("{usage}");
                    std::process::exit(2);
                }
            }
            i += 2;
        }
        if let Err(e) = cache::run_precompute(config, line, chunk) {
            eprintln!("precompute failed: {e}");
            std::process::exit(1);
        }
        return;
    }

    let registry: Registry = Arc::new(Mutex::new(HashMap::new()));
    let server = Server::http(ADDR).expect("failed to bind 127.0.0.1:3117");
    println!("wizgto-solver v{VERSION} listening on http://{ADDR}");
    println!("flop cache dir: {}", cache::cache_dir().display());

    for mut request in server.incoming_requests() {
        evict_expired(&registry);

        let mut body = String::new();
        let _ = request.as_reader().read_to_string(&mut body);

        let (status, payload) = if *request.method() == Method::Options {
            (204, None)
        } else {
            let url = request.url().to_string();
            let (code, value) = route(&registry, request.method(), &url, &body);
            (code, Some(value))
        };

        let mut response = Response::from_string(payload.map_or(String::new(), |v| v.to_string()))
            .with_status_code(status);
        for header in cors_headers() {
            response.add_header(header);
        }
        if status != 204 {
            response.add_header(
                Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap(),
            );
        }
        let _ = request.respond(response);
    }
}

fn cors_headers() -> Vec<Header> {
    vec![
        Header::from_bytes(&b"Access-Control-Allow-Origin"[..], &b"*"[..]).unwrap(),
        Header::from_bytes(&b"Access-Control-Allow-Headers"[..], &b"content-type"[..]).unwrap(),
        Header::from_bytes(
            &b"Access-Control-Allow-Methods"[..],
            &b"GET, POST, DELETE, OPTIONS"[..],
        )
        .unwrap(),
        Header::from_bytes(&b"Access-Control-Allow-Private-Network"[..], &b"true"[..]).unwrap(),
    ]
}

fn evict_expired(registry: &Registry) {
    let ttl = Duration::from_secs(SESSION_TTL_SECS);
    lock(registry).retain(|_, s| lock(s).created_at.elapsed() < ttl);
}

fn route(registry: &Registry, method: &Method, url: &str, body: &str) -> (u16, Value) {
    let (path, query) = url.split_once('?').unwrap_or((url, ""));
    let segments: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();

    match (method, segments.as_slice()) {
        (Method::Get, ["health"]) => (200, json!({"ok": true, "version": VERSION})),
        (Method::Post, ["session"]) => create_session(registry, body),
        (Method::Get, ["session", id]) => session_status(registry, id),
        (Method::Delete, ["session", id]) => delete_session(registry, id),
        (Method::Post, ["session", id, "event"]) => post_event(registry, id, body),
        (Method::Get, ["session", id, "strategy"]) => get_strategy(registry, id, query),
        _ => (404, json!({"error": "not found"})),
    }
}

fn err(status: u16, message: impl std::fmt::Display) -> (u16, Value) {
    (status, json!({"error": message.to_string()}))
}

fn find_session(registry: &Registry, id: &str) -> Option<SessionRef> {
    lock(registry).get(id).cloned()
}

// ---- POST /session ---------------------------------------------------------------------------

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct CreateSessionReq {
    board: Vec<String>,
    oop_range: String,
    ip_range: String,
    starting_pot: i32,
    effective_stack: i32,
    /// Optional: target exploitability as % of starting pot (clamped to 0.1-10, default 2.0).
    #[serde(default)]
    target_exploitability_pct: Option<f32>,
    /// Optional: max CFR iterations (clamped to 10-2000, default 300).
    #[serde(default)]
    max_iterations: Option<u32>,
    /// Optional: wall-clock solve budget in seconds (clamped to 5-300, default 20).
    #[serde(default)]
    max_solve_seconds: Option<f32>,
    /// Optional: skip the precomputed flop cache and always solve live
    /// (testing/debugging; cached sessions cover the flop street only).
    #[serde(default)]
    no_cache: Option<bool>,
}

fn create_session(registry: &Registry, body: &str) -> (u16, Value) {
    let req: CreateSessionReq = match serde_json::from_str(body) {
        Ok(r) => r,
        Err(e) => return err(400, format!("invalid request: {e}")),
    };
    // 3/4/5 board cards root the tree at the flop/turn/river.
    let initial_state = match req.board.len() {
        3 => BoardState::Flop,
        4 => BoardState::Turn,
        5 => BoardState::River,
        n => return err(400, format!("board must have 3, 4, or 5 cards, got {n}")),
    };
    let flop = match flop_from_str(&req.board[..3].join("")) {
        Ok(f) => f,
        Err(e) => return err(400, format!("bad board: {e}")),
    };
    let mut turn = NOT_DEALT;
    let mut river = NOT_DEALT;
    if let Some(card) = req.board.get(3) {
        turn = match card_from_str(card) {
            Ok(c) => c,
            Err(e) => return err(400, format!("bad turn card: {e}")),
        };
    }
    if let Some(card) = req.board.get(4) {
        river = match card_from_str(card) {
            Ok(c) => c,
            Err(e) => return err(400, format!("bad river card: {e}")),
        };
    }
    let mut board_cards: Vec<u8> = flop.to_vec();
    board_cards.extend([turn, river].iter().filter(|&&c| c != NOT_DEALT));
    board_cards.sort_unstable();
    if board_cards.windows(2).any(|w| w[0] == w[1]) {
        return err(400, "board contains duplicate cards");
    }
    let oop_range: Range = match req.oop_range.parse() {
        Ok(r) => r,
        Err(e) => return err(400, format!("bad oopRange: {e}")),
    };
    let ip_range: Range = match req.ip_range.parse() {
        Ok(r) => r,
        Err(e) => return err(400, format!("bad ipRange: {e}")),
    };
    if req.starting_pot <= 0 {
        return err(400, "startingPot must be positive");
    }
    if req.effective_stack <= 0 {
        return err(400, "effectiveStack must be positive");
    }
    let target_exploitability_pct = req
        .target_exploitability_pct
        .unwrap_or(DEFAULT_TARGET_EXPLOITABILITY_PCT)
        .clamp(
            TARGET_EXPLOITABILITY_PCT_RANGE.0,
            TARGET_EXPLOITABILITY_PCT_RANGE.1,
        );
    let max_iterations = req
        .max_iterations
        .unwrap_or(DEFAULT_MAX_ITERATIONS)
        .clamp(MAX_ITERATIONS_RANGE.0, MAX_ITERATIONS_RANGE.1);
    let max_solve_secs = req
        .max_solve_seconds
        .unwrap_or(DEFAULT_MAX_SOLVE_SECS)
        .clamp(MAX_SOLVE_SECS_RANGE.0, MAX_SOLVE_SECS_RANGE.1);

    let card_config = CardConfig {
        range: [oop_range, ip_range],
        flop,
        turn,
        river,
    };

    // Flop-rooted sessions may be served instantly from the precomputed cache.
    if initial_state == BoardState::Flop && !req.no_cache.unwrap_or(false) {
        if let Some(cached) = cache::find_cached(
            &req.oop_range,
            &req.ip_range,
            flop,
            req.starting_pot,
            req.effective_stack,
        ) {
            let exploitability_pct = cached.entry.exploitability_pct;
            let session = Arc::new(Mutex::new(Session {
                created_at: Instant::now(),
                status: Status::Ready,
                progress: 1.0,
                exploitability_pct,
                game: None,
                cached: Some(cached),
                pending_events: Vec::new(),
                approximated_path: false,
            }));
            let id = new_session_id();
            lock(registry).insert(id.clone(), session);
            println!("session {id}: served from flop cache");
            return (200, json!({"sessionId": id, "fromCache": true}));
        }
    }

    let session = Arc::new(Mutex::new(Session {
        created_at: Instant::now(),
        status: Status::Solving,
        progress: 0.0,
        exploitability_pct: f32::INFINITY,
        game: None,
        cached: None,
        pending_events: Vec::new(),
        approximated_path: false,
    }));

    let id = new_session_id();
    lock(registry).insert(id.clone(), Arc::clone(&session));
    spawn_solve(
        session,
        SolveSpec {
            card_config,
            initial_state,
            starting_pot: req.starting_pot,
            effective_stack: req.effective_stack,
            target_exploitability_pct,
            max_iterations,
            max_solve_secs,
        },
    );

    (200, json!({"sessionId": id}))
}

fn new_session_id() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    format!("{:x}-{:x}", millis, COUNTER.fetch_add(1, Ordering::Relaxed))
}

// ---- GET /session/{id} -------------------------------------------------------------------------

fn session_status(registry: &Registry, id: &str) -> (u16, Value) {
    let Some(session) = find_session(registry, id) else {
        return err(404, "unknown session");
    };
    let s = lock(&session);
    match &s.status {
        Status::Solving => (
            200,
            json!({"status": "solving", "progress": round4(s.progress.min(1.0))}),
        ),
        Status::Ready => match &s.cached {
            Some(cached) => (
                200,
                json!({
                    "status": "ready",
                    "exploitabilityPct": round4(s.exploitability_pct),
                    "fromCache": true,
                    "note": cached.note,
                }),
            ),
            None => (
                200,
                json!({"status": "ready", "exploitabilityPct": round4(s.exploitability_pct)}),
            ),
        },
        Status::Failed(e) => (200, json!({"status": "failed", "error": e})),
    }
}

// ---- DELETE /session/{id} ----------------------------------------------------------------------

fn delete_session(registry: &Registry, id: &str) -> (u16, Value) {
    match lock(registry).remove(id) {
        Some(_) => (200, json!({"ok": true})),
        None => err(404, "unknown session"),
    }
}

// ---- POST /session/{id}/event -------------------------------------------------------------------

fn post_event(registry: &Registry, id: &str, body: &str) -> (u16, Value) {
    let ev: EventReq = match serde_json::from_str(body) {
        Ok(e) => e,
        Err(e) => return err(400, format!("invalid event: {e}")),
    };
    let Some(session) = find_session(registry, id) else {
        return err(404, "unknown session");
    };
    let mut s = lock(&session);
    match &s.status {
        Status::Solving => {
            // Queue; applied in order once the solve completes.
            s.pending_events.push(ev);
            (200, json!({"applied": true, "approximated": false}))
        }
        Status::Failed(e) => err(409, format!("session failed: {e}")),
        Status::Ready => match apply_event(&mut s, &ev) {
            Ok(approximated) => (200, json!({"applied": true, "approximated": approximated})),
            Err((status, message)) => err(status, message),
        },
    }
}

// ---- GET /session/{id}/strategy -----------------------------------------------------------------

fn get_strategy(registry: &Registry, id: &str, query: &str) -> (u16, Value) {
    let params: HashMap<&str, &str> = query
        .split('&')
        .filter_map(|kv| kv.split_once('='))
        .collect();

    let Some(hand) = params.get("hand") else {
        return err(400, "missing \"hand\" query param");
    };
    let player = match params.get("player") {
        Some(&"oop") => 0usize,
        Some(&"ip") => 1usize,
        _ => return err(400, "\"player\" must be \"oop\" or \"ip\""),
    };
    if hand.len() != 4 {
        return err(400, "\"hand\" must be two cards, e.g. AhKd");
    }
    let (c1, c2) = match (card_from_str(&hand[0..2]), card_from_str(&hand[2..4])) {
        (Ok(a), Ok(b)) if a != b => (a, b),
        (Ok(_), Ok(_)) => return err(400, "hand cards must differ"),
        (Err(e), _) | (_, Err(e)) => return err(400, format!("bad hand: {e}")),
    };

    let Some(session) = find_session(registry, id) else {
        return err(404, "unknown session");
    };
    let mut s = lock(&session);
    match &s.status {
        Status::Solving => return err(409, "session is still solving"),
        Status::Failed(e) => return err(409, format!("session failed: {e}")),
        Status::Ready => {}
    }
    let approximated_path = s.approximated_path;

    if let Some(cached) = s.cached.as_ref() {
        if cached.done {
            return err(409, "flop street is over (cached session) — re-root at the current street");
        }
        if cached.real_flop.contains(&c1) || cached.real_flop.contains(&c2) {
            return err(404, "hand conflicts with the board");
        }
        let node = &cached.entry.nodes[cached.node_index[&cached.path]];
        if node.player as usize != player {
            return err(409, "not this player's turn");
        }
        let (m1, m2) = (cache::map_card(c1, &cached.perm), cache::map_card(c2, &cached.perm));
        let (lo, hi) = (m1.min(m2), m1.max(m2));
        let Some(hand_index) = cached.hands[player].iter().position(|&h| h == (lo, hi)) else {
            return err(404, "hand not in this player's range");
        };
        let entries: Vec<Value> = node
            .actions
            .iter()
            .map(|a| {
                let freq = round4(a.freqs[hand_index]);
                let ev = round2((a.evs[hand_index] as f64 * cached.scale) as f32);
                match a.amount {
                    Some(x) => {
                        json!({"action": a.action, "amount": cache::scale_amount(x, cached.scale), "freq": freq, "ev": ev})
                    }
                    None => json!({"action": a.action, "freq": freq, "ev": ev}),
                }
            })
            .collect();
        return (
            200,
            json!({
                "node": "flop",
                "actions": entries,
                "approximatedPath": approximated_path,
                "fromCache": true,
                "note": cached.note,
            }),
        );
    }

    let game = s.game.as_mut().expect("ready session has a game");

    if game.is_terminal_node() {
        return err(409, "hand is over (terminal node)");
    }
    if game.is_chance_node() {
        return err(409, "awaiting a turn/river card (chance node)");
    }
    if game.current_player() != player {
        return err(409, "not this player's turn");
    }

    let board = game.current_board();
    if board.contains(&c1) || board.contains(&c2) {
        return err(404, "hand conflicts with the board");
    }
    let street = match board.len() {
        3 => "flop",
        4 => "turn",
        _ => "river",
    };

    let hand_index = game
        .private_cards(player)
        .iter()
        .position(|&(a, b)| (a == c1 && b == c2) || (a == c2 && b == c1));
    let Some(hand_index) = hand_index else {
        return err(404, "hand not in this player's range");
    };
    let num_hands = game.private_cards(player).len();

    game.cache_normalized_weights();
    let strategy = game.strategy();
    let evs = game.expected_values_detail(player);
    let actions = game.available_actions();

    let entries: Vec<Value> = actions
        .iter()
        .enumerate()
        .map(|(i, action)| {
            let freq = round4(strategy[i * num_hands + hand_index]);
            let ev = round2(evs[i * num_hands + hand_index]);
            use postflop_solver::Action;
            let (name, amount) = match action {
                Action::Fold => ("fold", None),
                Action::Check => ("check", None),
                Action::Call => ("call", None),
                Action::Bet(x) => ("bet", Some(*x)),
                Action::Raise(x) => ("raise", Some(*x)),
                Action::AllIn(x) => ("allin", Some(*x)),
                _ => ("unknown", None),
            };
            match amount {
                Some(x) => json!({"action": name, "amount": x, "freq": freq, "ev": ev}),
                None => json!({"action": name, "freq": freq, "ev": ev}),
            }
        })
        .collect();

    (
        200,
        json!({
            "node": street,
            "actions": entries,
            "approximatedPath": approximated_path,
        }),
    )
}

fn round4(v: f32) -> f64 {
    (v as f64 * 10_000.0).round() / 10_000.0
}

fn round2(v: f32) -> f64 {
    (v as f64 * 100.0).round() / 100.0
}
