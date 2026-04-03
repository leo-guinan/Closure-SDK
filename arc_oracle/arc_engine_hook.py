"""
arc_engine_hook.py — Drop-in integration for arc_engine.py

HOW TO INTEGRATE
----------------
In arc_engine.py, add three lines:

    # At top of file:
    import sys
    sys.path.insert(0, '/path/to/arc_oracle')
    from arc_engine_hook import OracleHook

    # In ArcEngine.__init__:
    self.oracle_hook = OracleHook(game_id=self.game_id)

    # In ArcEngine.pre_turn (before Newton gate):
    oracle_result = self.oracle_hook.pre_turn(
        current_state=self._encode_current_state(),
        action_count=self.action_count,
        budget=self.budget,
        world_model_sigma=self.physics.mean_sigma(),
    )
    if oracle_result is not None:
        return oracle_result  # arc_engine executes this program directly

The hook handles:
  - episode start/end lifecycle
  - trigger logic (when to consult)
  - result execution and win recording
  - logging

STATE ENCODING
--------------
arc_engine must call self._encode_current_state() which returns a RelState.
Each game needs its own encoder. The hook provides helpers for each known game.
For unknown games it falls back to the frame-based encoder.

EXAMPLE INTEGRATION (wa30):
    def _encode_current_state(self) -> RelState:
        from arc_engine_hook import encode_wa30_live
        return encode_wa30_live(
            player_sprite=self.sensor.player,
            box_sprites=self.sensor.boxes,
            goal_positions=self.physics.goal_positions,
            level=self.current_level,
        )
"""

from __future__ import annotations

import os
import sys
from typing import Any, List, Optional, Tuple

from relational_encoder import (
    RelState, OracleBudget, encode_from_frame,
)
from normalized_encoder import (
    encode_wa30_normalized, encode_ls20_normalized,
    encode_r11l_normalized, encode_sc25_normalized,
    L3Constants, L3Registry, KNOWN_L3,
)
from runtime_oracle import RuntimeOracle, OraclePool, OracleTrigger, OracleResult
from offline_trainer import DNA_DIR

# ── Live state encoders ───────────────────────────────────────────────────────

def encode_wa30_live(
    player_x: float, player_y: float,
    player_rot: int,
    grabbed_box: Optional[Tuple[float, float]],
    box_positions: List[Tuple[float, float]],
    l3: Optional[L3Constants] = None,
    level: int = 0,
) -> RelState:
    """Encode wa30 live state with L3 normalization."""
    return encode_wa30_normalized(
        player_x=player_x, player_y=player_y,
        player_rot=player_rot, grabbed_box=grabbed_box,
        box_positions=box_positions,
        l3=l3, level=level,
    )


def encode_ls20_live(
    player_x: float, player_y: float,
    player_rot: int,
    pushbar_positions: List[Tuple[float, float]],
    l3: Optional[L3Constants] = None,
    level: int = 0,
) -> RelState:
    """Encode ls20 live state with L3 normalization."""
    return encode_ls20_normalized(
        player_x=player_x, player_y=player_y,
        player_rot=player_rot,
        pushbar_positions=pushbar_positions,
        l3=l3, level=level,
    )


def encode_r11l_live(
    active_node_x: float, active_node_y: float,
    other_nodes: List[Tuple[float, float]],
    l3: Optional[L3Constants] = None,
    level: int = 0,
) -> RelState:
    """Encode r11l live state with L3 normalization."""
    return encode_r11l_normalized(
        active_node_x=active_node_x, active_node_y=active_node_y,
        other_nodes=other_nodes,
        l3=l3, level=level,
    )


def encode_sc25_live(
    player_x: float, player_y: float,
    spell_selected: bool,
    l3: Optional[L3Constants] = None,
    level: int = 0,
) -> RelState:
    """Encode sc25 live state with L3 normalization."""
    return encode_sc25_normalized(
        player_x=player_x, player_y=player_y,
        spell_selected=spell_selected,
        l3=l3, level=level,
    )


# ── Hook ─────────────────────────────────────────────────────────────────────

class OracleHook:
    """
    Integrates the oracle into arc_engine's turn loop.

    Lifecycle:
        hook = OracleHook(game_id="wa30")
        hook.new_episode(episode=0, level=1)

        # Each turn (called from pre_turn):
        program = hook.pre_turn(state, action_count, budget, sigma)
        if program is not None:
            # Execute program actions in arc_engine
            won = execute_program(program)
            hook.post_program(won, start_state=state, actions_taken=program)

        # End of episode:
        hook.end_episode()
    """

    def __init__(
        self,
        game_id: str,
        dna_dir: str = DNA_DIR,
        sigma_drop_threshold: float = 0.8,
        verbose: bool = True,
    ):
        self.game_id = game_id
        self.oracle = RuntimeOracle(game_id, dna_dir)
        self.sigma_drop_threshold = sigma_drop_threshold
        self.verbose = verbose

        self._episode = 0
        self._level = 0
        self._action_count = 0
        self._episode_start_state: Optional[RelState] = None
        self._pending_program: Optional[List[Any]] = None
        self._sigma_drop_fired = False
        self._budget_half_fired = False

    def new_episode(self, episode: int, level: int):
        """Call at start of each episode."""
        self._episode = episode
        self._level = level
        self._action_count = 0
        self._episode_start_state = None
        self._pending_program = None
        self._sigma_drop_fired = False
        self._budget_half_fired = False
        self.oracle.new_episode(episode, level)
        if self.verbose:
            print(f"[oracle] Episode {episode} L{level+1} — budget {self.oracle.budget}")

    def pre_turn(
        self,
        state: RelState,
        action_count: int,
        budget: int,
        world_model_sigma: Optional[float] = None,
    ) -> Optional[List[Any]]:
        """
        Called before each turn. Returns a program (list of actions) or None.
        If a program is returned, arc_engine should execute ALL actions in it,
        not just the first one.
        """
        self._action_count = action_count

        # Determine trigger
        trigger = None

        if action_count == 0 and not self._episode_start_state:
            trigger = OracleTrigger.EPISODE_START
            self._episode_start_state = state

        elif (not self._sigma_drop_fired and
              world_model_sigma is not None and
              world_model_sigma > self.sigma_drop_threshold):
            trigger = OracleTrigger.SIGMA_DROP
            self._sigma_drop_fired = True

        elif (not self._budget_half_fired and
              budget > 0 and action_count >= budget // 2):
            trigger = OracleTrigger.BUDGET_HALF
            self._budget_half_fired = True

        if trigger is None:
            return None

        if not self.oracle.should_consult(trigger, action_count, budget, world_model_sigma):
            return None

        result = self.oracle.consult(state, trigger, action_count, budget, world_model_sigma)

        if self.verbose:
            print(f"[oracle] {trigger.name} → {result}")

        if result.hit:
            return result.actions

        return None

    def post_program(
        self,
        won: bool,
        start_state: RelState,
        actions_taken: List[Any],
    ):
        """
        Call after executing a program returned by pre_turn.
        Records outcome. If win: stores in DNA for future episodes.
        """
        if won:
            self.oracle.record_win(actions_taken, start_state)
            if self.verbose:
                print(f"[oracle] Win recorded — {len(actions_taken)} actions stored in DNA")
        else:
            if self.verbose:
                print(f"[oracle] Program failed after {len(actions_taken)} actions")

    def end_episode(self):
        """Call at end of each episode."""
        pass  # logs flushed automatically in new_episode

    def stats(self) -> dict:
        return self.oracle.stats()


# ── arc_engine patch helper ───────────────────────────────────────────────────

PATCH_TEMPLATE = '''
# === ARC ORACLE INTEGRATION ===
import sys as _sys
_sys.path.insert(0, '{oracle_dir}')
from arc_engine_hook import OracleHook, encode_wa30_live, encode_ls20_live, encode_r11l_live

# In ArcEngine.__init__, add:
#   self.oracle_hook = OracleHook(game_id=self.game_id)

# In ArcEngine.run_episode, before each action:
#   state = encode_{game_id}_live(...)
#   program = self.oracle_hook.pre_turn(state, action_count, budget, sigma)
#   if program:
#       won = self._execute_oracle_program(program)
#       self.oracle_hook.post_program(won, state, program)
#       continue  # skip normal action selection
# ==============================
'''


def print_integration_instructions(game_id: str):
    oracle_dir = os.path.dirname(os.path.abspath(__file__))
    print(PATCH_TEMPLATE.format(oracle_dir=oracle_dir, game_id=game_id))
