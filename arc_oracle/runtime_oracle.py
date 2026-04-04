"""
runtime_oracle.py — State -> DNA query -> action sequence or None

ROLE IN THE ENGINE
------------------
The oracle sits in arc_engine's pre_turn hook, budget-limited to 3 calls/episode.
It is consulted at three trigger points:

  TRIGGER_EPISODE_START:  first action of a new episode/level
  TRIGGER_SIGMA_DROP:     world model prediction confidence drops below threshold
  TRIGGER_BUDGET_HALF:    action count reaches 50% of budget

On each trigger:
  1. Encode current game state as RelState
  2. Query DNA: primary key (self→world, cross-game) + refinement (others, within-game)
  3. If sigma < MATCH_THRESHOLD: return the stored action sequence
  4. If no match: return None, world model handles this step

CALL BUDGET
-----------
Default: 3 calls per episode (ORACLE_CALLS_PER_EPISODE).
Each call either returns a program (use it) or None (fall through).
The oracle never retries within an episode after a miss — one miss per trigger.

PROGRAM EXECUTION
-----------------
When the oracle returns a program, arc_engine executes it directly:
  for action in oracle_program:
      obs = env.step(action)
      if obs.state == WIN: break
      if obs.state == GAME_OVER: break  # abort, let world model recover

If the program succeeds (level advances): log as oracle_win.
If the program fails midway: log partial actions in death_log,
resume world model from current state. Don't re-call oracle on same trigger.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

from relational_encoder import (
    RelState, OracleBudget, MATCH_THRESHOLD, ORACLE_CALLS_PER_EPISODE
)
from offline_trainer import GameDNA, OfflineTrainer, DNA_DIR
from normalized_encoder import CROSS_GAME_MATCH_THRESHOLD, WITHIN_GAME_MATCH_THRESHOLD

# ── Trigger types ────────────────────────────────────────────────────────────

class OracleTrigger(Enum):
    EPISODE_START  = auto()   # called at start of each episode
    SIGMA_DROP     = auto()   # world model confidence dropped
    BUDGET_HALF    = auto()   # action budget 50% consumed
    EXPLICIT       = auto()   # called manually by external code


@dataclass
class OracleResult:
    """What the oracle returns to arc_engine."""
    hit: bool                        # True = found a program, False = miss
    actions: List[Any]               # action sequence (empty if miss)
    sigma: float                     # how close the match was
    source: str                      # "confirmed_plan" | "bfs" | "live_win" | "miss"
    entry_note: str = ""             # human-readable from stored entry
    query_time_ms: float = 0.0

    @property
    def is_miss(self) -> bool:
        return not self.hit

    def __repr__(self):
        if self.hit:
            return (f"OracleResult(HIT σ={self.sigma:.3f} "
                    f"src={self.source} {len(self.actions)} actions)")
        return f"OracleResult(MISS σ={self.sigma:.3f})"


@dataclass
class OracleLog:
    """Per-episode oracle activity log."""
    episode: int = 0
    calls: List[dict] = field(default_factory=list)
    wins_from_oracle: int = 0
    misses: int = 0

    def record(self, trigger: OracleTrigger, result: OracleResult, action_count: int):
        self.calls.append({
            "trigger": trigger.name,
            "hit": result.hit,
            "sigma": result.sigma,
            "source": result.source,
            "actions_returned": len(result.actions),
            "action_count_at_call": action_count,
            "query_ms": result.query_time_ms,
        })
        if result.hit:
            pass  # win tracked externally
        else:
            self.misses += 1

    def to_dict(self) -> dict:
        return {
            "episode": self.episode,
            "calls": self.calls,
            "wins_from_oracle": self.wins_from_oracle,
            "misses": self.misses,
        }


# ── Oracle ────────────────────────────────────────────────────────────────────

class RuntimeOracle:
    """
    The oracle. Plugs into arc_engine pre_turn.

    Usage:
        oracle = RuntimeOracle(game_id="wa30")
        oracle.new_episode(episode_number, level)

        # In pre_turn:
        result = oracle.consult(current_state, trigger, action_count, budget)
        if result.hit:
            # execute result.actions directly
            pass
    """

    def __init__(
        self,
        game_id: str,
        dna_dir: str = DNA_DIR,
        match_threshold: float = MATCH_THRESHOLD,
        max_calls: int = ORACLE_CALLS_PER_EPISODE,
        log_path: Optional[str] = None,
    ):
        self.game_id = game_id
        self.dna = GameDNA(game_id, dna_dir)
        self.trainer = OfflineTrainer(dna_dir)
        self.threshold = match_threshold
        self.budget = OracleBudget(max_calls)
        self.log_path = log_path or os.path.join(dna_dir, f"{game_id}_oracle_log.json")

        self._episode_log = OracleLog()
        self._all_logs: List[dict] = []
        self._current_level: int = 0

        # Track which triggers have fired this episode (don't re-fire same trigger)
        self._fired_triggers: set = set()

        # Load existing logs
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path) as f:
                    self._all_logs = json.load(f)
            except Exception:
                self._all_logs = []

    def new_episode(self, episode: int, level: int):
        """Call at the start of each episode. Resets budget and trigger tracking."""
        if self._episode_log.calls:
            self._all_logs.append(self._episode_log.to_dict())
            self._save_logs()
        self._episode_log = OracleLog(episode=episode)
        self.budget.reset()
        self._fired_triggers = set()
        self._current_level = level

    def should_consult(
        self,
        trigger: OracleTrigger,
        action_count: int,
        budget: int,
        world_model_sigma: Optional[float] = None,
    ) -> bool:
        """
        Decide whether to call the oracle given current conditions.
        Returns False if: budget exhausted, trigger already fired, or conditions not met.
        """
        if not self.budget.can_call():
            return False
        if trigger in self._fired_triggers:
            return False

        if trigger == OracleTrigger.EPISODE_START:
            return action_count == 0

        if trigger == OracleTrigger.SIGMA_DROP:
            # Only consult if world model confidence is actually poor
            return world_model_sigma is not None and world_model_sigma > 0.8

        if trigger == OracleTrigger.BUDGET_HALF:
            return budget > 0 and action_count >= budget // 2

        if trigger == OracleTrigger.EXPLICIT:
            return True

        return False

    def consult(
        self,
        state: RelState,
        trigger: OracleTrigger,
        action_count: int,
        budget: int,
        world_model_sigma: Optional[float] = None,
    ) -> OracleResult:
        """
        Query DNA for a program matching the current state.
        Returns OracleResult (hit or miss).
        Budget is decremented on every call (hit or miss).
        """
        if not self.budget.record_call(trigger.name):
            return OracleResult(hit=False, actions=[], sigma=999.0, source="budget_exhausted")

        self._fired_triggers.add(trigger)
        t0 = time.monotonic()

        # Two-phase query:
        # Phase 1: within-game, full word (self+others+world), tight threshold
        # Phase 2: cross-game, self_word only, loose threshold (near-goal signal)
        hit = self.dna.query(
            state,
            level=self._current_level,
            threshold=WITHIN_GAME_MATCH_THRESHOLD,
            use_full_sigma=True,
        )

        # Cross-game fallback: self_word only, looser threshold
        # Only useful for near-goal states (clustering confirmed in normalized_encoder.py)
        # Always filter by level to prevent L1 plan being used for L2+
        if hit is None:
            hit = self.dna.query(
                state,
                level=self._current_level,
                threshold=CROSS_GAME_MATCH_THRESHOLD,
                use_full_sigma=False,
            )

        elapsed = (time.monotonic() - t0) * 1000

        if hit is not None:
            entry, sigma = hit
            result = OracleResult(
                hit=True,
                actions=entry.actions,
                sigma=sigma,
                source=entry.source,
                entry_note=entry.key.raw_note,
                query_time_ms=elapsed,
            )
        else:
            # Best sigma even on miss — useful for logging
            best_sigma = self._best_sigma(state)
            result = OracleResult(
                hit=False,
                actions=[],
                sigma=best_sigma,
                source="miss",
                query_time_ms=elapsed,
            )

        self._episode_log.record(trigger, result, action_count)
        return result

    def record_win(self, actions: List[Any], start_state: RelState):
        """
        Record a new win sequence discovered during live play.
        Stores in DNA so future episodes can use it.
        """
        self.trainer.store_live_win(
            self.game_id, start_state, actions, self._current_level
        )
        self.dna = GameDNA(self.game_id)  # reload
        self._episode_log.wins_from_oracle += 1

    def stats(self) -> dict:
        """Summary statistics across all logged episodes."""
        total_calls = sum(len(log["calls"]) for log in self._all_logs)
        total_hits = sum(
            sum(1 for c in log["calls"] if c["hit"])
            for log in self._all_logs
        )
        total_wins = sum(log["wins_from_oracle"] for log in self._all_logs)
        hit_rate = total_hits / total_calls if total_calls > 0 else 0.0

        return {
            "game_id": self.game_id,
            "dna_entries": len(self.dna),
            "episodes_logged": len(self._all_logs),
            "total_oracle_calls": total_calls,
            "total_hits": total_hits,
            "hit_rate": hit_rate,
            "total_oracle_wins": total_wins,
        }

    def _best_sigma(self, state: RelState) -> float:
        """Best sigma across all entries, regardless of threshold."""
        if not self.dna.entries:
            return 999.0
        return min(state.full_sigma(e.key) for e in self.dna.entries)

    def _save_logs(self):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, "w") as f:
            json.dump(self._all_logs, f, indent=2)


# ── Multi-game oracle pool ────────────────────────────────────────────────────

class OraclePool:
    """
    Manages one RuntimeOracle per game.
    arc_engine holds one OraclePool for the whole benchmark run.
    """

    def __init__(self, dna_dir: str = DNA_DIR):
        self.dna_dir = dna_dir
        self._oracles: Dict[str, RuntimeOracle] = {}

        # Auto-seed DNA tables from confirmed plans on first use
        trainer = OfflineTrainer(dna_dir)
        trainer.train_all(force=False)

    def get(self, game_id: str) -> RuntimeOracle:
        """Get or create oracle for a game."""
        if game_id not in self._oracles:
            self._oracles[game_id] = RuntimeOracle(game_id, self.dna_dir)
        return self._oracles[game_id]

    def consult(
        self,
        game_id: str,
        state: RelState,
        trigger: OracleTrigger,
        action_count: int,
        budget: int,
        world_model_sigma: Optional[float] = None,
    ) -> OracleResult:
        """Convenience: consult oracle for a specific game."""
        return self.get(game_id).consult(
            state, trigger, action_count, budget, world_model_sigma
        )

    def all_stats(self) -> List[dict]:
        return [oracle.stats() for oracle in self._oracles.values()]
