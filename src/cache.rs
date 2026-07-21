//! Precomputed flop-strategy cache.
//!
//! A cache entry stores, for one (OOP range, IP range, SPR) and one canonical
//! flop, the strategy (per-combo action frequencies + EVs) at every
//! flop-street decision node of a solved tree. A session that hits the cache
//! is ready instantly and answers flop strategy queries from the summary; the
//! extension re-roots at the turn (turn/river solves are sub-second), the same
//! path it already uses when a live solve isn't ready in time.
//!
//! Solves are normalized to starting pot 100 / stack = SPR x 100; amounts and
//! EVs scale linearly to the live pot. Suits are canonicalized: the flop is
//! mapped through the suit permutation that minimizes its card triple, and
//! lookups map hero hands through the same permutation (ranges are
//! suit-symmetric, so range strings are unaffected).
//!
//! Entries live in `server/cache/` as one JSON file per
//! `(range-hash, flop, spr)`; the `precompute` CLI mode fills the cache and is
//! resumable (existing files are skipped).

use crate::solver::*;
use postflop_solver::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

pub const CACHE_FORMAT_VERSION: u32 = 1;
/// A cache entry is usable when the live SPR is within this relative distance
/// of the solved SPR. Wide on purpose: with an SPR ladder per line (see
/// make-cache-config.mjs), a labeled ±25% match beats a multi-minute live
/// solve; the note reports both SPRs whenever they differ.
pub const SPR_REL_TOLERANCE: f64 = 0.25;

// ---- Suit canonicalization -------------------------------------------------------------------

/// All 24 permutations of the four suits (index = real suit, value = mapped suit).
fn suit_perms() -> Vec<[u8; 4]> {
    let mut out = Vec::with_capacity(24);
    let suits = [0u8, 1, 2, 3];
    for &a in &suits {
        for &b in &suits {
            if b == a {
                continue;
            }
            for &c in &suits {
                if c == a || c == b {
                    continue;
                }
                let d = 6 - a - b - c;
                out.push([a, b, c, d]);
            }
        }
    }
    out
}

pub fn map_card(card: u8, perm: &[u8; 4]) -> u8 {
    (card & !3) | perm[(card & 3) as usize]
}

/// Canonical form of a flop: the suit permutation (real suit -> canonical
/// suit) whose image, sorted descending, is lexicographically smallest.
/// Deterministic, so precompute and lookup always agree — including which of
/// several equivalent permutations maps off-board suits.
pub fn canonical_flop(cards: [u8; 3]) -> ([u8; 3], [u8; 4]) {
    let mut best: Option<([u8; 3], [u8; 4])> = None;
    for perm in suit_perms() {
        let mut mapped = cards.map(|c| map_card(c, &perm));
        mapped.sort_unstable_by(|a, b| b.cmp(a));
        if best.is_none() || mapped < best.unwrap().0 {
            best = Some((mapped, perm));
        }
    }
    best.unwrap()
}

pub fn flop_string(cards: [u8; 3]) -> String {
    cards.iter().map(|&c| card_to_string(c).unwrap_or_default()).collect()
}

/// FNV-1a over the two range strings; the cache key ignores SPR (checked
/// against the entry's stored value at lookup time).
pub fn range_hash(oop_range: &str, ip_range: &str) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in oop_range.bytes().chain([b'|']).chain(ip_range.bytes()) {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("{h:016x}")
}

/// Cache directory: `../../cache` relative to the binary
/// (`server/target/release/wizgto-solver` -> `server/cache`), overridable via
/// `WIZGTO_CACHE_DIR`. launchd runs the binary with no useful cwd, so this
/// must not be cwd-relative.
pub fn cache_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("WIZGTO_CACHE_DIR") {
        return PathBuf::from(dir);
    }
    std::env::current_exe()
        .ok()
        .and_then(|exe| exe.ancestors().nth(3).map(|p| p.join("cache")))
        .unwrap_or_else(|| PathBuf::from("cache"))
}

// ---- Entry format ----------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CacheEntry {
    pub version: u32,
    pub oop_range: String,
    pub ip_range: String,
    /// Solved at starting pot `pot`, effective stack `spr * pot`.
    pub spr: f64,
    pub pot: i32,
    pub flop: String,
    pub exploitability_pct: f32,
    /// Combo strings (canonical suits) in solver order, per player.
    pub oop_hands: Vec<String>,
    pub ip_hands: Vec<String>,
    pub nodes: Vec<CacheNode>,
}

#[derive(Serialize, Deserialize)]
pub struct CacheNode {
    /// Action-token path from the flop root, e.g. "check/bet:50".
    pub path: String,
    /// 0 = OOP, 1 = IP (player to act).
    pub player: u8,
    pub actions: Vec<CacheAction>,
}

#[derive(Serialize, Deserialize)]
pub struct CacheAction {
    pub action: String,
    /// Bet/raise-to/all-in total in normalized units.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub amount: Option<i32>,
    /// Per-hand action frequency, indexed like the player's hands list.
    pub freqs: Vec<f32>,
    /// Per-hand EV in normalized units.
    pub evs: Vec<f32>,
}

fn action_token(a: &Action) -> (String, Option<i32>) {
    match a {
        Action::Fold => ("fold".into(), None),
        Action::Check => ("check".into(), None),
        Action::Call => ("call".into(), None),
        Action::Bet(x) => ("bet".into(), Some(*x)),
        Action::Raise(x) => ("raise".into(), Some(*x)),
        Action::AllIn(x) => ("allin".into(), Some(*x)),
        _ => ("unknown".into(), None),
    }
}

fn token_string(action: &str, amount: Option<i32>) -> String {
    match amount {
        Some(x) => format!("{action}:{x}"),
        None => action.to_string(),
    }
}

// ---- Summary extraction ----------------------------------------------------------------------

/// Walks every flop-street decision node of a solved game and records the
/// strategy + EVs for the player to act. Stops at chance (turn) and terminal
/// nodes — later streets are the extension's re-root path.
pub fn extract_flop_nodes(game: &mut PostFlopGame) -> Vec<CacheNode> {
    let mut nodes = Vec::new();
    walk(game, String::new(), &mut nodes);
    game.back_to_root();
    nodes
}

fn walk(game: &mut PostFlopGame, path: String, out: &mut Vec<CacheNode>) {
    if game.is_terminal_node() || game.is_chance_node() {
        return;
    }
    let player = game.current_player();
    let actions = game.available_actions();
    let num_hands = game.private_cards(player).len();
    game.cache_normalized_weights();
    let strategy = game.strategy();
    let evs = game.expected_values_detail(player);

    let cache_actions: Vec<CacheAction> = actions
        .iter()
        .enumerate()
        .map(|(i, a)| {
            let (action, amount) = action_token(a);
            CacheAction {
                action,
                amount,
                freqs: strategy[i * num_hands..(i + 1) * num_hands]
                    .iter()
                    .map(|v| (v * 10_000.0).round() / 10_000.0)
                    .collect(),
                evs: evs[i * num_hands..(i + 1) * num_hands]
                    .iter()
                    .map(|v| (v * 100.0).round() / 100.0)
                    .collect(),
            }
        })
        .collect();
    out.push(CacheNode { path: path.clone(), player: player as u8, actions: cache_actions });

    let history = game.history().to_vec();
    for (i, a) in actions.iter().enumerate() {
        let (action, amount) = action_token(a);
        let child = if path.is_empty() {
            token_string(&action, amount)
        } else {
            format!("{path}/{}", token_string(&action, amount))
        };
        game.play(i);
        walk(game, child, out);
        game.apply_history(&history);
    }
}

// ---- Lookup ----------------------------------------------------------------------------------

/// Runtime state of a session served from the cache.
pub struct CachedState {
    pub entry: CacheEntry,
    /// path -> index into entry.nodes.
    pub node_index: HashMap<String, usize>,
    /// Parsed hands per player, matching entry.{oop,ip}_hands order.
    pub hands: [Vec<(u8, u8)>; 2],
    pub path: String,
    /// real suit -> canonical suit.
    pub perm: [u8; 4],
    /// live pot / normalized pot: scales amounts and EVs.
    pub scale: f64,
    pub real_flop: [u8; 3],
    /// No decisions left in the cached street (folded, called, or checked through).
    pub done: bool,
    pub note: String,
}

fn parse_hands(strings: &[String]) -> Vec<(u8, u8)> {
    strings
        .iter()
        .map(|s| {
            let a = card_from_str(&s[0..2]).unwrap_or(0);
            let b = card_from_str(&s[2..4]).unwrap_or(0);
            (a.min(b), a.max(b))
        })
        .collect()
}

/// Probes the cache for `(ranges, flop)` at a compatible SPR.
pub fn find_cached(
    oop_range: &str,
    ip_range: &str,
    real_flop: [u8; 3],
    starting_pot: i32,
    effective_stack: i32,
) -> Option<CachedState> {
    let dir = cache_dir();
    let hash = range_hash(oop_range, ip_range);
    let (canon, perm) = canonical_flop(real_flop);
    let prefix = format!("{hash}-{}-spr", flop_string(canon));
    let spr_req = effective_stack as f64 / starting_pot as f64;

    let mut best: Option<(f64, PathBuf)> = None;
    for f in fs::read_dir(&dir).ok()?.flatten() {
        let name = f.file_name().to_string_lossy().into_owned();
        if !name.starts_with(&prefix) || !name.ends_with(".json") {
            continue;
        }
        let spr: f64 = name[prefix.len()..name.len() - 5].parse().ok()?;
        let dist = (spr_req / spr).ln().abs();
        if dist <= (1.0 + SPR_REL_TOLERANCE).ln() && best.as_ref().is_none_or(|(d, _)| dist < *d) {
            best = Some((dist, f.path()));
        }
    }
    let (_, path) = best?;
    let entry: CacheEntry = serde_json::from_str(&fs::read_to_string(path).ok()?).ok()?;
    if entry.version != CACHE_FORMAT_VERSION
        || entry.oop_range != oop_range
        || entry.ip_range != ip_range
    {
        return None;
    }

    let mut note = "precomputed flop cache".to_string();
    if (spr_req / entry.spr - 1.0).abs() > 0.02 {
        note = format!("precomputed flop cache (solved at SPR {:.1}; this pot {:.1})", entry.spr, spr_req);
    }
    Some(CachedState {
        node_index: entry.nodes.iter().enumerate().map(|(i, n)| (n.path.clone(), i)).collect(),
        hands: [parse_hands(&entry.oop_hands), parse_hands(&entry.ip_hands)],
        path: String::new(),
        perm,
        scale: starting_pot as f64 / entry.pot as f64,
        real_flop,
        done: false,
        note,
        entry,
    })
}

pub fn scale_amount(amount: i32, scale: f64) -> i32 {
    (amount as f64 * scale).round() as i32
}

/// Applies an observed flop action to a cached session by matching it against
/// the current node's actions (nearest size for bet/raise, like the live
/// tree). Card events are refused: the extension re-roots at the turn.
/// Returns `approximated` like the live path.
pub fn apply_cached_event(state: &mut CachedState, ev: &EventReq) -> Result<bool, (u16, String)> {
    if state.done {
        return Err((409, "flop street is over (cached session) — re-root at the current street".into()));
    }
    if ev.kind == "card" {
        return Err((409, "cached session covers the flop only — re-root at the current street".into()));
    }
    if ev.kind != "action" {
        return Err((400, format!("unknown event kind \"{}\"", ev.kind)));
    }
    let action = ev.action.as_ref().ok_or((400u16, "missing \"action\"".to_string()))?;
    let node = &state.entry.nodes[state.node_index[&state.path]];

    let (token, approximated) = match action.as_str() {
        "check" | "call" | "fold" => {
            let a = node
                .actions
                .iter()
                .find(|a| a.action == *action)
                .ok_or((409u16, format!("{action} is not available at this node")))?;
            (token_string(&a.action, a.amount), false)
        }
        "bet" | "raise" => {
            let amount = ev.amount.ok_or((400u16, "missing \"amount\"".to_string()))?;
            if amount <= 0 {
                return Err((400, "\"amount\" must be positive".to_string()));
            }
            let mut best: Option<(&CacheAction, i32)> = None;
            for a in &node.actions {
                let kind_matches = matches!(
                    (action.as_str(), a.action.as_str()),
                    ("bet", "bet") | ("bet", "allin") | ("raise", "raise") | ("raise", "allin")
                );
                if !kind_matches {
                    continue;
                }
                let real = scale_amount(a.amount.unwrap_or(0), state.scale);
                if best.is_none_or(|(_, b)| (real - amount).abs() < (b - amount).abs()) {
                    best = Some((a, real));
                }
            }
            let (a, mapped) = best.ok_or((409u16, format!("no {action} is available at this node")))?;
            let approximated = (mapped - amount).abs() as f64 / amount as f64 > APPROX_REL_DIFF;
            (token_string(&a.action, a.amount), approximated)
        }
        other => return Err((400, format!("unknown action \"{other}\""))),
    };

    state.path = if state.path.is_empty() { token } else { format!("{}/{}", state.path, token) };
    // No stored node down this branch means no flop decisions remain
    // (fold, call, check-through, or all-in called).
    if !state.node_index.contains_key(&state.path) {
        state.done = true;
    }
    Ok(approximated)
}

// ---- Precompute ------------------------------------------------------------------------------

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PrecomputeLine {
    pub label: String,
    pub oop_range: String,
    pub ip_range: String,
    pub spr: f64,
    /// Solve the top N canonical flops by frequency weight (default: all 1755).
    #[serde(default)]
    pub flops: Option<usize>,
    /// Explicit flops (e.g. for tests); overrides `flops` when present.
    #[serde(default)]
    pub flop_list: Option<Vec<String>>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PrecomputeConfig {
    pub lines: Vec<PrecomputeLine>,
}

/// All canonical flops with their frequency weights (how many of the 22100
/// raw flops map to each), sorted by weight descending — precomputing in this
/// order maximizes hit rate for a partial run.
pub fn canonical_flops_by_weight() -> Vec<([u8; 3], u32)> {
    let mut weights: HashMap<[u8; 3], u32> = HashMap::new();
    for a in 0..52u8 {
        for b in (a + 1)..52 {
            for c in (b + 1)..52 {
                let (canon, _) = canonical_flop([a, b, c]);
                *weights.entry(canon).or_insert(0) += 1;
            }
        }
    }
    let mut out: Vec<_> = weights.into_iter().collect();
    out.sort_unstable_by(|x, y| y.1.cmp(&x.1).then(y.0.cmp(&x.0)));
    out
}

/// `line_filter` restricts to one config line; `chunk = (i, k)` takes every
/// k-th flop starting at i (interleaved, so each chunk stays frequency-
/// ordered). Both exist so CI runners can split the batch into parallel jobs.
pub fn run_precompute(
    config_path: &str,
    line_filter: Option<usize>,
    chunk: Option<(usize, usize)>,
) -> Result<(), String> {
    let config: PrecomputeConfig =
        serde_json::from_str(&fs::read_to_string(config_path).map_err(|e| e.to_string())?)
            .map_err(|e| format!("bad config: {e}"))?;
    if let Some((i, k)) = chunk {
        if k == 0 || i >= k {
            return Err(format!("bad --chunk {i}/{k}: need 0 <= i < k"));
        }
    }
    let dir = cache_dir();
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    println!("precompute: cache dir {}", dir.display());

    let all_flops = canonical_flops_by_weight();
    println!("precompute: {} canonical flops (of 22100 raw)", all_flops.len());

    for (line_index, line) in config.lines.iter().enumerate() {
        if line_filter.is_some_and(|n| n != line_index) {
            continue;
        }
        let hash = range_hash(&line.oop_range, &line.ip_range);
        let mut flops: Vec<[u8; 3]> = match &line.flop_list {
            Some(list) => list
                .iter()
                .map(|s| flop_from_str(s).map(|f| canonical_flop(f).0))
                .collect::<Result<_, _>>()
                .map_err(|e| format!("bad flopList entry: {e}"))?,
            None => all_flops
                .iter()
                .take(line.flops.unwrap_or(usize::MAX))
                .map(|(f, _)| *f)
                .collect(),
        };
        if let Some((i, k)) = chunk {
            flops = flops
                .into_iter()
                .enumerate()
                .filter(|(idx, _)| idx % k == i)
                .map(|(_, f)| f)
                .collect();
        }
        println!(
            "\n=== [line {line_index}{}] {} — {} flops, SPR {}, hash {hash}",
            chunk.map_or(String::new(), |(i, k)| format!(", chunk {i}/{k}")),
            line.label,
            flops.len(),
            line.spr
        );

        let pot = 100;
        let stack = (line.spr * pot as f64).round() as i32;
        let (mut solved, mut skipped) = (0u32, 0u32);
        let started = Instant::now();
        for (i, canon) in flops.iter().enumerate() {
            let name = format!("{hash}-{}-spr{}.json", flop_string(*canon), line.spr);
            let path = dir.join(&name);
            if path.exists() {
                skipped += 1;
                continue;
            }
            match solve_one(line, *canon, pot, stack) {
                Ok(entry) => {
                    let tmp = path.with_extension("json.tmp");
                    fs::write(&tmp, serde_json::to_string(&entry).map_err(|e| e.to_string())?)
                        .map_err(|e| e.to_string())?;
                    fs::rename(&tmp, &path).map_err(|e| e.to_string())?;
                    solved += 1;
                    println!(
                        "  [{}/{}] {} solved ({:.1}% expl), {:.0}s elapsed total",
                        i + 1,
                        flops.len(),
                        flop_string(*canon),
                        entry.exploitability_pct,
                        started.elapsed().as_secs_f64(),
                    );
                }
                Err(e) => println!("  [{}/{}] {} FAILED: {e}", i + 1, flops.len(), flop_string(*canon)),
            }
        }
        println!("=== {}: {solved} solved, {skipped} already cached", line.label);
    }
    Ok(())
}

/// Solve one canonical flop at the line's normalized pot/stack and summarize
/// its flop street. Same tree/accuracy settings as live sessions
/// (see solver.rs), minus the session plumbing.
fn solve_one(
    line: &PrecomputeLine,
    canon: [u8; 3],
    pot: i32,
    stack: i32,
) -> Result<CacheEntry, String> {
    let oop: Range = line.oop_range.parse()?;
    let ip: Range = line.ip_range.parse()?;
    let mut flop_sorted = canon;
    flop_sorted.sort_unstable();
    let card_config = CardConfig {
        range: [oop, ip],
        flop: flop_sorted,
        turn: NOT_DEALT,
        river: NOT_DEALT,
    };
    let action_tree = ActionTree::new(tree_config(BoardState::Flop, pot, stack)?)?;
    let mut game = PostFlopGame::with_config(card_config, action_tree)?;
    let (mem_uncompressed, _) = game.memory_usage();
    game.allocate_memory(mem_uncompressed > MAX_UNCOMPRESSED_BYTES);

    let target = pot as f32 * DEFAULT_TARGET_EXPLOITABILITY_PCT / 100.0;
    let mut exploitability = compute_exploitability(&game);
    for t in 0..DEFAULT_MAX_ITERATIONS {
        if exploitability <= target {
            break;
        }
        solve_step(&game, t);
        if (t + 1) % 10 == 0 || t + 1 == DEFAULT_MAX_ITERATIONS {
            exploitability = compute_exploitability(&game);
        }
    }
    finalize(&mut game);

    let oop_hands = holes_to_strings(game.private_cards(0)).map_err(|e| e.to_string())?;
    let ip_hands = holes_to_strings(game.private_cards(1)).map_err(|e| e.to_string())?;
    let nodes = extract_flop_nodes(&mut game);
    Ok(CacheEntry {
        version: CACHE_FORMAT_VERSION,
        oop_range: line.oop_range.clone(),
        ip_range: line.ip_range.clone(),
        spr: line.spr,
        pot,
        flop: flop_string(canon),
        exploitability_pct: 100.0 * exploitability / pot as f32,
        oop_hands,
        ip_hands,
        nodes,
    })
}

// ---- Tests -----------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn cards(s: &str) -> [u8; 3] {
        flop_from_str(s).unwrap()
    }

    #[test]
    fn there_are_1755_canonical_flops() {
        assert_eq!(canonical_flops_by_weight().len(), 1755);
    }

    #[test]
    fn suit_permuted_flops_share_a_canonical_form() {
        // Same ranks, suits permuted every which way.
        let variants = ["Td9h6c", "Th9d6c", "Ts9c6d", "Tc9s6h", "Td9c6h"];
        let forms: Vec<_> = variants.iter().map(|v| canonical_flop(cards(v)).0).collect();
        assert!(forms.windows(2).all(|w| w[0] == w[1]), "forms differ: {forms:?}");
        // Different texture (two-tone) must not collide.
        let two_tone = canonical_flop(cards("Td9d6c")).0;
        assert_ne!(two_tone, forms[0]);
    }

    #[test]
    fn perm_maps_the_real_flop_onto_the_canonical_flop() {
        for s in ["Td9h6c", "AsKs2s", "7h7d2c", "QdJd8c", "5c5d5h"] {
            let real = cards(s);
            let (canon, perm) = canonical_flop(real);
            let mut mapped = real.map(|c| map_card(c, &perm));
            mapped.sort_unstable_by(|a, b| b.cmp(a));
            assert_eq!(mapped, canon, "flop {s}");
        }
    }

    #[test]
    fn range_hash_is_order_sensitive() {
        assert_ne!(range_hash("AA", "KK"), range_hash("KK", "AA"));
    }
}
