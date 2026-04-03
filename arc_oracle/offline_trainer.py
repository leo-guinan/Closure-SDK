"""
offline_trainer.py — Build DNA tables from confirmed plans and offline BFS

WHAT IT DOES
------------
For each confirmed plan, stores a (start_state_word, action_sequence) pair in a
JSON-backed DNA table. At runtime the oracle queries by state word and retrieves
the nearest stored program.

WHY JSON NOT RUST DNA
---------------------
The Rust DNA engine requires pyo3 bindings which aren't compiled on this machine.
We use a JSON store with the same query interface: encode state as RelWord,
compute word_sigma against all stored keys, return nearest below threshold.
When pyo3 bindings are available this swaps to VerificationWord + WordMemory
with zero changes to the caller.

DATA FORMAT
-----------
Each DNA table is a JSON file per game:
  arc_oracle/dna/{game_id}.json
  {
    "game_id": "wa30",
    "entries": [
      {
        "key": { "self_word": ..., "others_word": ..., "world_word": ... },
        "level": 1,
        "actions": [1,1,3,...],
        "source": "confirmed_plan",
        "sigma_at_store": 0.0
      },
      ...
    ]
  }

CONFIRMED PLANS (from arc_engine.py / skill memory)
-----------------------------------------------------
  wa30 L1: [1,1,3,1,1,1,3,3,5,4,4,4,5,1,4,4,5,2,3,3,5,2,5,1,1,5]
  ls20 L1: [3,3,3,1,1,1,1,4,4,4,1,1,1]
  ls20 L2: [1,4,1,1,1,1,1,4,4,2,4,2,2,2,2,2,2,2,3,3,4,1,4,1,2,1,1,1,1,1,1,1,3,3,3,3,3,3,2,3,2,2,2,2,2]
  ls20 L3: [1,1,1,1,1,1,1,1,3,2,2,2,2,2,3,3,4,4,2,2,2,1,1,1,1,1,1,4,2,2,4,4,4,4,1,1,1,3,1,2,2,4,2,2,2,2,2,2,2]
  ls20 L4: [3,3,3,2,2,2,3,2,2,3,3,1,2,1,2,1,2,1,1,3,3,1,2,3,3,1,1,1,2,2,4,1,1,1,1,4,1,4,1,1,3,3,3]
  r11l L0: [(6,(4,16)),(6,(16,16)),...] — click-only, stored as action tuples
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from relational_encoder import (
    RelState, RelWord, RelCell,
    encode_wa30_state, encode_ls20_state, encode_r11l_state,
    MATCH_THRESHOLD,
)

DNA_DIR = os.path.join(os.path.dirname(__file__), "dna")

# ── Confirmed plans ──────────────────────────────────────────────────────────

# Start states derived from game mechanics (see arc_engine.py + skill)
# wa30 grid_step=4, player starts at top-left area
_WA30_L1_START = dict(
    player_x=8, player_y=8, player_rot=0, grabbed_box=None,
    box_positions=[(16, 8), (8, 20), (20, 20)],
    goal_positions=[(24, 8), (24, 12), (24, 16)],
)
_WA30_L1_PLAN = [1,1,3,1,1,1,3,3,5,4,4,4,5,1,4,4,5,2,3,3,5,2,5,1,1,5]

# ls20 grid_step=4, goal = exit tile (approximate positions from confirmed plans)
_LS20_L1_START = dict(
    player_x=4, player_y=4, player_rot=0,
    goal_positions=[(44, 4)],
    pushbar_positions=[],
)
_LS20_L1_PLAN = [3,3,3,1,1,1,1,4,4,4,1,1,1]

_LS20_L2_START = dict(
    player_x=4, player_y=4, player_rot=0,
    goal_positions=[(44, 44)],
    pushbar_positions=[(24, 12)],
)
_LS20_L2_PLAN = [1,4,1,1,1,1,1,4,4,2,4,2,2,2,2,2,2,2,3,3,4,1,4,1,2,1,1,1,1,1,1,1,3,3,3,3,3,3,2,3,2,2,2,2,2]

_LS20_L3_START = dict(
    player_x=4, player_y=4, player_rot=0,
    goal_positions=[(44, 44)],
    pushbar_positions=[(12, 24), (36, 24)],
)
_LS20_L3_PLAN = [1,1,1,1,1,1,1,1,3,2,2,2,2,2,3,3,4,4,2,2,2,1,1,1,1,1,1,4,2,2,4,4,4,4,1,1,1,3,1,2,2,4,2,2,2,2,2,2,2]

_LS20_L4_START = dict(
    player_x=4, player_y=4, player_rot=0,
    goal_positions=[(44, 44)],
    pushbar_positions=[(8, 16), (16, 16), (24, 16), (32, 16),
                       (8, 32), (16, 32), (24, 32), (32, 32)],
)
_LS20_L4_PLAN = [3,3,3,2,2,2,3,2,2,3,3,1,2,1,2,1,2,1,1,3,3,1,2,3,3,1,1,1,2,2,4,1,1,1,1,4,1,4,1,1,3,3,3]

# r11l L0: active node at (6,19), target at (36,18)
_R11L_L0_START = dict(
    active_node_x=6, active_node_y=19,
    target_x=36, target_y=18,
    other_nodes=[(25, 57)],
)
# r11l actions are (action_id, (x, y)) tuples for A6 clicks
_R11L_L0_PLAN = [
    (6, (4,16)), (6, (16,16)), (6, (4,12)), (6, (48,28)), (6, (10,5)),
    (6, (4,12)), (6, (0,32)),  (6, (63,24)),(6, (23,24)), (6, (3,0)),
    (6, (32,8)), (6, (29,61)), (6, (52,28)),
]


# ── Storage ──────────────────────────────────────────────────────────────────

@dataclass
class DNAEntry:
    key: RelState
    level: int
    actions: List[Any]          # List[int] for movement, List[tuple] for click
    source: str                 # "confirmed_plan" | "bfs" | "live_win"
    sigma_at_store: float = 0.0

    def to_dict(self) -> dict:
        return {
            "key": self.key.to_json(),
            "level": self.level,
            "actions": self.actions,
            "source": self.source,
            "sigma_at_store": self.sigma_at_store,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DNAEntry":
        return cls(
            key=RelState.from_json(d["key"]),
            level=d["level"],
            actions=d["actions"],
            source=d["source"],
            sigma_at_store=d.get("sigma_at_store", 0.0),
        )


class GameDNA:
    """JSON-backed DNA table for one game."""

    def __init__(self, game_id: str, dna_dir: str = DNA_DIR):
        self.game_id = game_id
        self.path = os.path.join(dna_dir, f"{game_id}.json")
        self.entries: List[DNAEntry] = []
        os.makedirs(dna_dir, exist_ok=True)
        if os.path.exists(self.path):
            self._load()

    def _load(self):
        with open(self.path) as f:
            data = json.load(f)
        self.entries = [DNAEntry.from_dict(e) for e in data.get("entries", [])]

    def save(self):
        with open(self.path, "w") as f:
            json.dump({
                "game_id": self.game_id,
                "entries": [e.to_dict() for e in self.entries],
            }, f, indent=2)

    def store(self, state: RelState, actions: List[Any],
              level: int, source: str = "confirmed_plan"):
        entry = DNAEntry(key=state, level=level, actions=actions, source=source)
        self.entries.append(entry)
        self.save()
        return entry

    def query(
        self,
        query_state: RelState,
        level: Optional[int] = None,
        threshold: float = MATCH_THRESHOLD,
        use_full_sigma: bool = False,
    ) -> Optional[Tuple[DNAEntry, float]]:
        """
        Find nearest stored entry to query_state.

        Returns (entry, sigma) if sigma < threshold, else None.
        use_full_sigma=False: primary key only (cross-game comparable)
        use_full_sigma=True: all three words (within-game precision)
        """
        best_entry = None
        best_sigma = float("inf")

        for entry in self.entries:
            if level is not None and entry.level != level:
                continue
            if use_full_sigma:
                s = query_state.full_sigma(entry.key)
            else:
                s = query_state.primary_sigma(entry.key)
            if s < best_sigma:
                best_sigma = s
                best_entry = entry

        if best_entry is not None and best_sigma < threshold:
            return best_entry, best_sigma
        return None

    def __len__(self):
        return len(self.entries)

    def __repr__(self):
        return f"GameDNA({self.game_id}, {len(self.entries)} entries)"


# ── Trainer ──────────────────────────────────────────────────────────────────

class OfflineTrainer:
    """
    Seeds DNA tables from confirmed plans.
    Call train_all() once. Tables persist across sessions.
    """

    def __init__(self, dna_dir: str = DNA_DIR):
        self.dna_dir = dna_dir
        self.trained: Dict[str, GameDNA] = {}

    def train_all(self, force: bool = False) -> Dict[str, int]:
        """Train all confirmed plans. Returns {game_id: entries_added}."""
        added = {}
        added["wa30"] = self._train_wa30(force)
        added["ls20"] = self._train_ls20(force)
        added["r11l"] = self._train_r11l(force)
        return added

    def _train_wa30(self, force: bool = False) -> int:
        dna = GameDNA("wa30", self.dna_dir)
        if len(dna) > 0 and not force:
            return 0

        state = encode_wa30_state(level=1, **_WA30_L1_START)
        dna.store(state, _WA30_L1_PLAN, level=1, source="confirmed_plan")
        self.trained["wa30"] = dna
        return 1

    def _train_ls20(self, force: bool = False) -> int:
        dna = GameDNA("ls20", self.dna_dir)
        if len(dna) >= 4 and not force:
            return 0

        plans = [
            (1, _LS20_L1_START, _LS20_L1_PLAN),
            (2, _LS20_L2_START, _LS20_L2_PLAN),
            (3, _LS20_L3_START, _LS20_L3_PLAN),
            (4, _LS20_L4_START, _LS20_L4_PLAN),
        ]
        count = 0
        for level, start, plan in plans:
            state = encode_ls20_state(level=level, **start)
            dna.store(state, plan, level=level, source="confirmed_plan")
            count += 1
        self.trained["ls20"] = dna
        return count

    def _train_r11l(self, force: bool = False) -> int:
        dna = GameDNA("r11l", self.dna_dir)
        if len(dna) > 0 and not force:
            return 0

        state = encode_r11l_state(level=0, **_R11L_L0_START)
        dna.store(state, _R11L_L0_PLAN, level=0, source="confirmed_plan")
        self.trained["r11l"] = dna
        return 1

    def store_live_win(
        self,
        game_id: str,
        start_state: RelState,
        actions: List[Any],
        level: int,
    ) -> DNAEntry:
        """
        Store a win discovered during live play.
        Called by the oracle when a new win sequence is observed.
        """
        dna = GameDNA(game_id, self.dna_dir)
        entry = dna.store(start_state, actions, level=level, source="live_win")
        return entry

    def store_bfs_solution(
        self,
        game_id: str,
        start_state: RelState,
        actions: List[Any],
        level: int,
    ) -> DNAEntry:
        """Store a solution found by offline BFS."""
        dna = GameDNA(game_id, self.dna_dir)
        entry = dna.store(start_state, actions, level=level, source="bfs")
        return entry

    def get_dna(self, game_id: str) -> GameDNA:
        if game_id not in self.trained:
            self.trained[game_id] = GameDNA(game_id, self.dna_dir)
        return self.trained[game_id]


# ── BFS trajectory harvester ──────────────────────────────────────────────────

def harvest_bfs_trajectories(
    game_id: str,
    env_path: str,
    encoder_fn,
    trainer: OfflineTrainer,
    max_levels: int = 9,
    verbose: bool = True,
) -> int:
    """
    Run offline BFS on a game using its local environment files.
    Stores each found solution in DNA.

    env_path: path to the game's environment directory
              e.g. /root/arc-engine/environment_files/wa30/...
    encoder_fn: callable(game_instance, level_index) -> RelState

    Returns number of solutions stored.
    """
    import importlib.util
    import sys
    import copy

    # Find the game module
    game_files = [f for f in os.listdir(env_path) if f.endswith('.py') and not f.startswith('_')]
    if not game_files:
        if verbose:
            print(f"[harvest_bfs] No .py files in {env_path}")
        return 0

    game_file = os.path.join(env_path, game_files[0])
    spec = importlib.util.spec_from_file_location(f"{game_id}_env", game_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    stored = 0
    for level_idx in range(max_levels):
        try:
            game = getattr(mod, game_id.split('-')[0].upper())()
        except Exception:
            # Try lowercase
            cls_name = game_id.split('-')[0].replace('_', '').capitalize()
            try:
                game = getattr(mod, cls_name)()
            except Exception as e:
                if verbose:
                    print(f"[harvest_bfs] Can't instantiate game: {e}")
                break

        try:
            GameAction = mod.GameAction
            from arcengine.enums import ActionInput
            game.perform_action(ActionInput(id=GameAction.RESET))
            if level_idx > 0:
                # Apply solutions for prior levels to reach this level state
                pass  # TODO: apply prior level solutions
        except Exception as e:
            if verbose:
                print(f"[harvest_bfs] Level {level_idx} setup failed: {e}")
            break

        # Encode start state
        try:
            start_state = encoder_fn(game, level_idx)
        except Exception as e:
            if verbose:
                print(f"[harvest_bfs] Encoder failed at level {level_idx}: {e}")
            continue

        # Simple BFS — game-agnostic, uses deepcopy
        actions_found = _simple_bfs(game, mod, level_idx, max_steps=60, verbose=verbose)
        if actions_found is not None:
            trainer.store_bfs_solution(game_id, start_state, actions_found, level_idx)
            stored += 1
            if verbose:
                print(f"[harvest_bfs] {game_id} L{level_idx+1}: {len(actions_found)} actions stored")

    return stored


def _simple_bfs(game, mod, level_idx: int, max_steps: int = 60, verbose: bool = False) -> Optional[List]:
    """
    Deepcopy BFS on a game instance. Returns action sequence or None.
    Game-agnostic: tries all available actions until win or budget exhausted.
    """
    try:
        from arcengine.enums import ActionInput
        GameAction = mod.GameAction
        GameState = mod.GameState if hasattr(mod, 'GameState') else None
    except Exception:
        return None

    available = [GameAction.ACTION1, GameAction.ACTION2,
                 GameAction.ACTION3, GameAction.ACTION4, GameAction.ACTION5]

    import copy
    from collections import deque

    # State key — use frame hash if available
    def get_key(g):
        try:
            return str(g._current_level_index) + str(id(g))
        except Exception:
            return str(id(g))

    queue = deque([(copy.deepcopy(game), [])])
    visited = set()
    nodes = 0

    while queue:
        g, path = queue.popleft()
        nodes += 1

        if nodes > 5000:
            break

        if len(path) >= max_steps:
            continue

        for action in available:
            try:
                g2 = copy.deepcopy(g)
                result = g2.perform_action(InputAction(id=action))

                # Check win
                won = False
                if hasattr(g2, 'qgzorkgosv') and g2.qgzorkgosv:
                    won = True
                elif GameState and hasattr(result, 'state') and result and \
                     hasattr(result.state, 'WIN') and result.state == GameState.WIN:
                    won = True
                elif hasattr(g2, '_levels_completed') and g2._levels_completed > level_idx:
                    won = True

                if won:
                    return path + [action.value if hasattr(action, 'value') else action]

                key = get_key(g2)
                if key not in visited:
                    visited.add(key)
                    queue.append((g2, path + [action.value if hasattr(action, 'value') else action]))
            except Exception:
                continue

    return None


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Seed DNA tables from confirmed plans")
    parser.add_argument("--force", action="store_true", help="Re-seed even if tables exist")
    parser.add_argument("--dna-dir", default=DNA_DIR, help="DNA storage directory")
    parser.add_argument("--show", action="store_true", help="Show stored entries")
    args = parser.parse_args()

    trainer = OfflineTrainer(dna_dir=args.dna_dir)
    result = trainer.train_all(force=args.force)

    print("Seeded DNA tables:")
    for game_id, count in result.items():
        dna = trainer.get_dna(game_id)
        print(f"  {game_id}: {count} new entries ({len(dna)} total)")
        if args.show:
            for i, entry in enumerate(dna.entries):
                note = entry.key.raw_note
                print(f"    [{i}] L{entry.level} {entry.source} "
                      f"{len(entry.actions)} actions  {note}")
