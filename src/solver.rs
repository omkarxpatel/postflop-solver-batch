//! Session model and solver logic wrapping postflop-solver.
//!
//! One session per hand (heads-up). The full flop->river tree is solved once on a
//! background thread; observed events then walk the solved tree.

use crate::cache::{apply_cached_event, CachedState};
use postflop_solver::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::panic::AssertUnwindSafe;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

// ---- Solver settings (see docs/SOLVER_API.md) ----------------------------------------------

/// Flop bet sizes for both players: 50% of the pot.
pub const FLOP_BET_SIZES: &str = "50%";
/// Turn bet sizes for both players: 75% of the pot.
pub const TURN_BET_SIZES: &str = "75%";
/// River bet sizes for both players: 75% of the pot.
pub const RIVER_BET_SIZES: &str = "75%";
/// Raise sizing: raise to 3.5x the previous bet.
/// (Contract originally suggested ~2.7x; widened to 3.5x to shorten raise chains so a
/// full-accuracy flop solve of a deep-stacked pot stays under ~60s on an M-series Mac.)
pub const RAISE_SIZES: &str = "3.5x";
/// Add an all-in action when (maximum bet size) <= 1.5x pot (geometric all-in threshold).
pub const ADD_ALLIN_THRESHOLD: f64 = 1.5;
/// Force all-in (replace raises) when SPR after the opponent's call <= 0.5.
/// (Contract originally suggested ~0.15; raised to 0.5 to trim the deepest raise levels —
/// same ~60s solve-time budget as RAISE_SIZES.)
pub const FORCE_ALLIN_THRESHOLD: f64 = 0.5;
/// Merge bet actions with close sizes (PioSOLVER algorithm), threshold 0.1.
pub const MERGING_THRESHOLD: f64 = 0.1;
/// Default maximum number of CFR iterations (overridable per session via `maxIterations`).
pub const DEFAULT_MAX_ITERATIONS: u32 = 300;
/// Clamp range for the per-session `maxIterations` override.
pub const MAX_ITERATIONS_RANGE: (u32, u32) = (10, 2000);
/// Default target exploitability as % of the starting pot (overridable per session via
/// `targetExploitabilityPct`). Relaxed from 0.5 to 2.0 for live play.
pub const DEFAULT_TARGET_EXPLOITABILITY_PCT: f32 = 2.0;
/// Live sessions stop solving after this wall-clock budget even above the
/// exploitability target — advice within ~20s beats perfection after 2min
/// (wide heads-up ranges can be brutally slow). Overridable per session via
/// `maxSolveSeconds`; the precompute batch does NOT use this (quality first).
pub const DEFAULT_MAX_SOLVE_SECS: f32 = 20.0;
/// Clamp range for the per-session `maxSolveSeconds` override.
pub const MAX_SOLVE_SECS_RANGE: (f32, f32) = (5.0, 300.0);
/// Clamp range for the per-session `targetExploitabilityPct` override.
pub const TARGET_EXPLOITABILITY_PCT_RANGE: (f32, f32) = (0.1, 10.0);
/// Use 16-bit compressed solver storage when the uncompressed estimate exceeds this (1 GiB).
pub const MAX_UNCOMPRESSED_BYTES: u64 = 1 << 30;
/// Sessions older than this are evicted (30 minutes).
pub const SESSION_TTL_SECS: u64 = 30 * 60;
/// Bet/raise mapping: flag `approximated` when the relative size difference exceeds 20%.
pub const APPROX_REL_DIFF: f64 = 0.20;

// ---- Session model --------------------------------------------------------------------------

pub enum Status {
    Solving,
    Ready,
    Failed(String),
}

pub struct Session {
    pub created_at: Instant,
    pub status: Status,
    /// Solve progress in [0, 1].
    pub progress: f32,
    /// Latest known exploitability as % of the starting pot.
    pub exploitability_pct: f32,
    /// Present once status is Ready.
    pub game: Option<PostFlopGame>,
    /// Present when the session was served from the precomputed flop cache
    /// (status is Ready immediately and `game` stays None).
    pub cached: Option<CachedState>,
    /// Events received while solving; applied in order once ready.
    pub pending_events: Vec<EventReq>,
    /// True if any applied bet/raise was size-approximated.
    pub approximated_path: bool,
}

pub type SessionRef = Arc<Mutex<Session>>;
pub type Registry = Arc<Mutex<HashMap<String, SessionRef>>>;

/// Locks a mutex, recovering from poisoning (a panicked solver thread must not kill the server).
pub fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

#[derive(Debug, Clone, Deserialize)]
pub struct EventReq {
    pub kind: String,           // "action" | "card"
    pub action: Option<String>, // "check" | "bet" | "raise" | "call" | "fold"
    pub amount: Option<i32>,    // for bet: total bet; for raise: raise-to total
    pub card: Option<String>,   // for kind "card"
}

// ---- Solving --------------------------------------------------------------------------------

/// Everything the background solve thread needs, resolved from the create-session request.
pub struct SolveSpec {
    pub card_config: CardConfig,
    /// Root street of the tree, matching the board length (3/4/5 -> Flop/Turn/River).
    pub initial_state: BoardState,
    pub starting_pot: i32,
    pub effective_stack: i32,
    /// Target exploitability as % of the starting pot (already clamped).
    pub target_exploitability_pct: f32,
    /// Maximum CFR iterations (already clamped).
    pub max_iterations: u32,
    /// Wall-clock solve budget in seconds (already clamped).
    pub max_solve_secs: f32,
}

/// Spawns the background solve thread for a freshly created session.
pub fn spawn_solve(session: SessionRef, spec: SolveSpec) {
    std::thread::spawn(move || {
        let result = std::panic::catch_unwind(AssertUnwindSafe(|| solve_session(&session, spec)));
        let error = match result {
            Ok(Ok(())) => return,
            Ok(Err(e)) => e,
            Err(_) => "solver thread panicked".to_string(),
        };
        let mut s = lock(&session);
        s.status = Status::Failed(error);
    });
}

pub fn tree_config(
    initial_state: BoardState,
    starting_pot: i32,
    effective_stack: i32,
) -> Result<TreeConfig, String> {
    let flop = BetSizeOptions::try_from((FLOP_BET_SIZES, RAISE_SIZES))?;
    let turn = BetSizeOptions::try_from((TURN_BET_SIZES, RAISE_SIZES))?;
    let river = BetSizeOptions::try_from((RIVER_BET_SIZES, RAISE_SIZES))?;
    Ok(TreeConfig {
        initial_state,
        starting_pot,
        effective_stack,
        rake_rate: 0.0,
        rake_cap: 0.0,
        flop_bet_sizes: [flop.clone(), flop],
        turn_bet_sizes: [turn.clone(), turn],
        river_bet_sizes: [river.clone(), river],
        turn_donk_sizes: None,
        river_donk_sizes: None,
        add_allin_threshold: ADD_ALLIN_THRESHOLD,
        force_allin_threshold: FORCE_ALLIN_THRESHOLD,
        merging_threshold: MERGING_THRESHOLD,
    })
}

fn solve_session(session: &SessionRef, spec: SolveSpec) -> Result<(), String> {
    let build_start = Instant::now();
    let action_tree = ActionTree::new(tree_config(
        spec.initial_state,
        spec.starting_pot,
        spec.effective_stack,
    )?)?;
    let mut game = PostFlopGame::with_config(spec.card_config, action_tree)?;

    let (mem_uncompressed, mem_compressed) = game.memory_usage();
    let compress = mem_uncompressed > MAX_UNCOMPRESSED_BYTES;
    game.allocate_memory(compress);
    println!(
        "solve: {:?}-rooted tree built in {:.1}s, memory estimate {:.0}MB uncompressed / {:.0}MB compressed, using {}",
        spec.initial_state,
        build_start.elapsed().as_secs_f64(),
        mem_uncompressed as f64 / (1024.0 * 1024.0),
        mem_compressed as f64 / (1024.0 * 1024.0),
        if compress { "compressed" } else { "uncompressed" },
    );

    let solve_start = Instant::now();
    let target = spec.starting_pot as f32 * spec.target_exploitability_pct / 100.0;
    let mut exploitability = compute_exploitability(&game);
    let mut iterations = 0;

    for t in 0..spec.max_iterations {
        if exploitability <= target {
            break;
        }
        if solve_start.elapsed().as_secs_f32() > spec.max_solve_secs {
            println!(
                "solve: wall-clock budget ({:.0}s) hit at {:.2}% exploitability — serving as-is",
                spec.max_solve_secs,
                100.0 * exploitability / spec.starting_pot as f32,
            );
            break;
        }
        solve_step(&game, t);
        iterations = t + 1;
        if (t + 1) % 10 == 0 || t + 1 == spec.max_iterations {
            exploitability = compute_exploitability(&game);
        }
        let mut s = lock(session);
        s.progress = (t + 1) as f32 / spec.max_iterations as f32;
        s.exploitability_pct = 100.0 * exploitability / spec.starting_pot as f32;
    }

    finalize(&mut game);
    println!(
        "solve: {} iterations in {:.1}s, exploitability {:.4}% of pot (target {:.4}%)",
        iterations,
        solve_start.elapsed().as_secs_f64(),
        100.0 * exploitability / spec.starting_pot as f32,
        spec.target_exploitability_pct,
    );

    let mut s = lock(session);
    s.progress = 1.0;
    s.exploitability_pct = 100.0 * exploitability / spec.starting_pot as f32;
    s.game = Some(game);
    s.status = Status::Ready;

    // Apply events that queued up while solving, in order.
    let pending = std::mem::take(&mut s.pending_events);
    for ev in &pending {
        if let Err((_, msg)) = apply_event(&mut s, ev) {
            s.status = Status::Failed(format!("queued event could not be applied: {msg}"));
            break;
        }
    }
    Ok(())
}

// ---- Event application ----------------------------------------------------------------------

/// Applies an observed event to a ready session's game tree.
/// Returns `approximated` on success, or `(http_status, error)` on failure.
pub fn apply_event(s: &mut Session, ev: &EventReq) -> Result<bool, (u16, String)> {
    if let Some(cached) = s.cached.as_mut() {
        let approximated = apply_cached_event(cached, ev)?;
        if approximated {
            s.approximated_path = true;
        }
        return Ok(approximated);
    }
    let game = s
        .game
        .as_mut()
        .ok_or((409u16, "session not ready".to_string()))?;

    if game.is_terminal_node() {
        return Err((409, "hand is over (terminal node)".to_string()));
    }

    match ev.kind.as_str() {
        "card" => {
            if !game.is_chance_node() {
                return Err((409, "not awaiting a card at this node".to_string()));
            }
            let card_str = ev
                .card
                .as_ref()
                .ok_or((400u16, "missing \"card\"".to_string()))?;
            let card = card_from_str(card_str).map_err(|e| (400u16, e))?;
            if game.possible_cards() & (1u64 << card) == 0 {
                return Err((409, format!("card {card_str} cannot be dealt here")));
            }
            game.play(card as usize);
            Ok(false)
        }
        "action" => {
            if game.is_chance_node() {
                return Err((409, "awaiting a turn/river card (chance node)".to_string()));
            }
            let action = ev
                .action
                .as_ref()
                .ok_or((400u16, "missing \"action\"".to_string()))?;
            let available = game.available_actions();
            match action.as_str() {
                "check" | "call" | "fold" => {
                    let want = match action.as_str() {
                        "check" => Action::Check,
                        "call" => Action::Call,
                        _ => Action::Fold,
                    };
                    let idx = available
                        .iter()
                        .position(|a| *a == want)
                        .ok_or((409u16, format!("{action} is not available at this node")))?;
                    game.play(idx);
                    Ok(false)
                }
                "bet" | "raise" => {
                    let amount = ev
                        .amount
                        .ok_or((400u16, "missing \"amount\"".to_string()))?;
                    if amount <= 0 {
                        return Err((400, "\"amount\" must be positive".to_string()));
                    }
                    // Map to the nearest tree action of the same kind (all-in counts for both;
                    // raise amounts are raise-to totals in both the tree and the event).
                    let mut best: Option<(usize, i32)> = None;
                    for (i, a) in available.iter().enumerate() {
                        let total = match (action.as_str(), a) {
                            ("bet", Action::Bet(x)) => *x,
                            ("bet", Action::AllIn(x)) => *x,
                            ("raise", Action::Raise(x)) => *x,
                            ("raise", Action::AllIn(x)) => *x,
                            _ => continue,
                        };
                        if best.is_none_or(|(_, b)| (total - amount).abs() < (b - amount).abs()) {
                            best = Some((i, total));
                        }
                    }
                    let (idx, mapped) = best
                        .ok_or((409u16, format!("no {action} is available at this node")))?;
                    let approximated =
                        (mapped - amount).abs() as f64 / amount as f64 > APPROX_REL_DIFF;
                    game.play(idx);
                    if approximated {
                        s.approximated_path = true;
                    }
                    Ok(approximated)
                }
                other => Err((400, format!("unknown action \"{other}\""))),
            }
        }
        other => Err((400, format!("unknown event kind \"{other}\""))),
    }
}
