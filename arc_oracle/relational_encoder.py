"""
relational_encoder.py — Self / Others / World word encoder for ARC-AGI-3

WALTER'S UPDATE (c8366a3)
--------------------------
Walter added VerificationCell / VerificationWord / VerificationArithmetic to the VM
with full DNA persistence via WordMemory. Our RelCell/RelWord here mirror that
structure exactly:

  Walter's VerificationCell  →  our RelCell (plane_axis, phase, turns)
  Walter's VerificationWord  →  our RelWord (list of RelCells)
  Walter's WordMemory.save   →  our GameDNA.store (JSON now, WordMemory later)

When pyo3 bindings ship, the swap is:
  RelCell(axis, phase, turns) → VerificationCell.from_phase_and_turns(plane, phase, turns)
  RelWord.word_sigma()        → VerificationArithmetic.subtract_words() + sigma on result
  GameDNA.store()             → WordMemory.save_word(table, word_id, word)
  GameDNA.query()             → WordMemory.load_word() + word-distance scan

KEY NEW PRIMITIVE: subtract_words + sigma
Walter's VerificationArithmetic.subtract_words(a, b) computes the geometric difference
as a VerificationWord where each cell's phase = a_phase - b_phase and turns track
borrowing. sigma of the result word's geometry() gives a proper S³ distance.
This is strictly better than our current sum-of-cell-sigmas approximation.
We document the upgrade path here but use the Python approximation until bindings exist.


DESIGN
------
A game state is encoded as three VerificationWords:

  self_word   (3 cells)   — where am I relative to the goal?
  others_word (3N cells)  — where are the N active entities relative to me and the goal?
  world_word  (3 cells)   — what does the goal/background look like right now?

Each cell is a (plane, phase, turns) triple stored as a float triple.
We don't import the Rust VM here — cells are plain Python dataclasses.
The Rust VM loads them at query/store time via word_memory.

CROSS-GAME TRANSFER
-------------------
self_word encodes NORMALIZED distance and direction to goal.
Normalization uses max_dist discovered by L3 WorldConstants (or sqrt(w²+h²) if unknown).
After normalization:
  - states near the goal have small dist_phase regardless of which game
  - states mid-task with carry=True cluster together regardless of which game
  - this makes the self_word a valid cross-game primary query key

others_word is game-specific — entity count and roles vary.
world_word is game-specific — goal pixel mask varies.
Both are used as refinement keys within a game, not for cross-game retrieval.

QUERY PROTOCOL
--------------
1. Encode current state -> (self_word, others_word, world_word)
2. Query DNA with search_composite([self_key, others_key])
3. If sigma < MATCH_THRESHOLD: retrieve stored program, execute it
4. Else: return None (fall through to world model)

Budget: oracle is called at most ORACLE_CALLS_PER_EPISODE times per episode.
Default: 3. (episode start, first confidence drop, budget halfway point)
"""

from __future__ import annotations

import math
import json
import struct
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ── Constants ────────────────────────────────────────────────────────────────

GRID_W = 64
GRID_H = 64
MAX_DIST = math.sqrt(GRID_W ** 2 + GRID_H ** 2)  # ~90.5 pixels

ORACLE_CALLS_PER_EPISODE = 3
MATCH_THRESHOLD = 0.4  # sigma radians — tune after first real runs

# ── Cell ─────────────────────────────────────────────────────────────────────

@dataclass
class RelCell:
    """One cell of a relational word.

    plane_axis: which Euler plane (0=i, 1=j, 2=k)
    phase:      rotation angle in [0, 2π]
    turns:      completed full cycles (twist sheet — 0=Direct, 1=Inverted, ...)
    """
    plane_axis: int   # 0, 1, or 2
    phase: float      # radians [0, 2π)
    turns: int        # completed cycles

    def to_bytes(self) -> bytes:
        return struct.pack(">Bfd", self.plane_axis, self.phase, float(self.turns))

    @classmethod
    def from_bytes(cls, b: bytes) -> "RelCell":
        axis, phase, turns = struct.unpack(">Bfd", b)
        return cls(axis, phase, int(turns))

    def geometry(self) -> Tuple[float, float, float, float]:
        """Unit quaternion [w, x, y, z] for this cell."""
        w = math.cos(self.phase / 2)
        xyz = [0.0, 0.0, 0.0]
        xyz[self.plane_axis] = math.sin(self.phase / 2)
        return (w, xyz[0], xyz[1], xyz[2])

    @staticmethod
    def sigma_between(a: "RelCell", b: "RelCell") -> float:
        """Geodesic distance between two cells on S³."""
        qa = a.geometry()
        qb = b.geometry()
        # gap = qa^-1 * qb
        # For same-axis cells this simplifies to phase difference
        if a.plane_axis == b.plane_axis:
            diff = abs(a.phase - b.phase) % (2 * math.pi)
            if diff > math.pi:
                diff = 2 * math.pi - diff
            return math.acos(max(-1.0, min(1.0, math.cos(diff / 2))))
        # Cross-axis: full hamilton product
        w1, x1, y1, z1 = qa
        w2, x2, y2, z2 = qb
        # inverse(qa) = [w1, -x1, -y1, -z1]
        gw = w1*w2 + x1*x2 + y1*y2 + z1*z2
        return math.acos(max(-1.0, min(1.0, abs(gw))))


@dataclass
class RelWord:
    """A list of RelCells encoding one semantic layer."""
    cells: List[RelCell] = field(default_factory=list)

    def word_sigma(self, other: "RelWord") -> float:
        """Sum of per-cell sigmas. Pads missing cells with identity (phase=0)."""
        n = max(len(self.cells), len(other.cells))
        total = 0.0
        for i in range(n):
            ca = self.cells[i] if i < len(self.cells) else RelCell(0, 0.0, 0)
            cb = other.cells[i] if i < len(other.cells) else RelCell(0, 0.0, 0)
            total += RelCell.sigma_between(ca, cb)
        return total

    def to_flat(self) -> List[float]:
        """Flat float list for DNA storage: [axis, phase, turns, axis, phase, turns, ...]"""
        out = []
        for c in self.cells:
            out.extend([float(c.plane_axis), c.phase, float(c.turns)])
        return out

    @classmethod
    def from_flat(cls, data: List[float]) -> "RelWord":
        assert len(data) % 3 == 0
        cells = []
        for i in range(0, len(data), 3):
            cells.append(RelCell(int(data[i]), data[i+1], int(data[i+2])))
        return cls(cells)

    def to_json(self) -> dict:
        return {"cells": [{"axis": c.plane_axis, "phase": c.phase, "turns": c.turns}
                          for c in self.cells]}

    @classmethod
    def from_json(cls, d: dict) -> "RelWord":
        return cls([RelCell(c["axis"], c["phase"], c["turns"]) for c in d["cells"]])


@dataclass
class RelState:
    """Complete relational encoding of one game state."""
    self_word:   RelWord   # 3 cells: dist_to_goal, dir_to_goal, carry
    others_word: RelWord   # 3N cells: per-entity (dist_to_self, dist_to_goal, role)
    world_word:  RelWord   # 3 cells: goal_dist_norm, goal_visible, background_stability

    game_id:     str = ""
    level:       int = 0
    raw_note:    str = ""  # human-readable for debugging

    def primary_sigma(self, other: "RelState") -> float:
        """Cross-game comparable distance: self_word only."""
        return self.self_word.word_sigma(other.self_word)

    def full_sigma(self, other: "RelState") -> float:
        """Within-game distance: all three words."""
        return (self.self_word.word_sigma(other.self_word) +
                self.others_word.word_sigma(other.others_word) +
                self.world_word.word_sigma(other.world_word))

    def to_json(self) -> dict:
        return {
            "self_word": self.self_word.to_json(),
            "others_word": self.others_word.to_json(),
            "world_word": self.world_word.to_json(),
            "game_id": self.game_id,
            "level": self.level,
            "raw_note": self.raw_note,
        }

    @classmethod
    def from_json(cls, d: dict) -> "RelState":
        return cls(
            self_word=RelWord.from_json(d["self_word"]),
            others_word=RelWord.from_json(d["others_word"]),
            world_word=RelWord.from_json(d["world_word"]),
            game_id=d.get("game_id", ""),
            level=d.get("level", 0),
            raw_note=d.get("raw_note", ""),
        )


# ── Encoding helpers ─────────────────────────────────────────────────────────

def _phase(value: float, max_value: float) -> float:
    """Normalize scalar to [0, π]. 0 = at goal / zero, π = maximum departure."""
    if max_value <= 0:
        return 0.0
    return min(1.0, abs(value) / max_value) * math.pi


def _dir_phase(dx: float, dy: float) -> float:
    """Direction to target as phase in [0, π].
    
    We use abs(atan2) → [0, π] instead of full [0, 2π] to avoid
    wrap-around discontinuity at ±π that would inflate cross-game sigma.
    Two states pointing SW and SE look similar (both angled down), which
    is the right behavior for approach heuristics.
    """
    angle = math.atan2(abs(dy), abs(dx))  # [0, π/2]
    return angle * 2  # stretch to [0, π]


def encode_movement_game(
    player_x: float, player_y: float,
    player_rot: int,          # 0=UP, 90=RIGHT, 180=DOWN, 270=LEFT
    carrying: bool,
    goal_positions: List[Tuple[float, float]],   # all goal tile centers
    entity_positions: List[Tuple[float, float, bool]],  # (x, y, is_active)
    max_dist: float = MAX_DIST,
    game_id: str = "",
    level: int = 0,
) -> RelState:
    """
    Encode a movement-based game state (wa30, ls20, tr87, g50t).

    Self word (3 cells, i/j/k planes):
      cell 0 (i): normalized distance from player to nearest goal
      cell 1 (j): direction angle to nearest goal
      cell 2 (k): carry state (0=not carrying, π=carrying)

    Others word (3 cells per entity):
      cell 3N+0 (i): entity dist to player (normalized)
      cell 3N+1 (j): entity dist to nearest goal (normalized)
      cell 3N+2 (k): entity active/dangerous flag

    World word (3 cells):
      cell 0 (i): mean dist of all goals to player (global goal density)
      cell 1 (j): rotation state of player
      cell 2 (k): always 0 for movement games (no world-phase concept yet)
    """
    if not goal_positions:
        nearest_goal_dist = max_dist
        nearest_dx, nearest_dy = max_dist, max_dist
        mean_goal_dist = max_dist
    else:
        dists = [(math.sqrt((gx - player_x)**2 + (gy - player_y)**2), gx - player_x, gy - player_y)
                 for gx, gy in goal_positions]
        dists.sort()
        nearest_goal_dist, nearest_dx, nearest_dy = dists[0]
        mean_goal_dist = sum(d[0] for d in dists) / len(dists)

    self_cells = [
        RelCell(0, _phase(nearest_goal_dist, max_dist), 0),   # dist to goal on i
        RelCell(1, _dir_phase(nearest_dx, nearest_dy), 0),    # direction on j
        RelCell(2, math.pi if carrying else 0.0, 0),          # carry on k
    ]

    others_cells = []
    for ex, ey, active in entity_positions:
        edist_player = math.sqrt((ex - player_x)**2 + (ey - player_y)**2)
        if goal_positions:
            edist_goal = min(math.sqrt((ex - gx)**2 + (ey - gy)**2)
                             for gx, gy in goal_positions)
        else:
            edist_goal = max_dist
        others_cells.extend([
            RelCell(0, _phase(edist_player, max_dist), 0),
            RelCell(1, _phase(edist_goal, max_dist), 0),
            RelCell(2, math.pi if active else 0.0, 0),
        ])

    rot_phase = (player_rot / 360.0) * math.pi
    world_cells = [
        RelCell(0, _phase(mean_goal_dist, max_dist), 0),
        RelCell(1, rot_phase, 0),
        RelCell(2, 0.0, 0),
    ]

    return RelState(
        self_word=RelWord(self_cells),
        others_word=RelWord(others_cells),
        world_word=RelWord(world_cells),
        game_id=game_id,
        level=level,
        raw_note=f"player=({player_x},{player_y}) rot={player_rot} carry={carrying}",
    )


def encode_click_game(
    active_node_x: float, active_node_y: float,
    target_x: float, target_y: float,
    other_nodes: List[Tuple[float, float]],  # other routing node positions
    max_dist: float = MAX_DIST,
    game_id: str = "",
    level: int = 0,
) -> RelState:
    """
    Encode a click-only game state (r11l, vc33, ft09, tn36).

    'self'  = active routing node or primary interactive element
    'goal'  = target position the ball/element must reach
    'others' = other routing nodes / interactive elements
    """
    dx = target_x - active_node_x
    dy = target_y - active_node_y
    dist = math.sqrt(dx*dx + dy*dy)

    self_cells = [
        RelCell(0, _phase(dist, max_dist), 0),
        RelCell(1, _dir_phase(dx, dy), 0),
        RelCell(2, 0.0, 0),  # no carry concept in click games
    ]

    others_cells = []
    for nx, ny in other_nodes:
        ndist = math.sqrt((nx - active_node_x)**2 + (ny - active_node_y)**2)
        ntarget_dist = math.sqrt((nx - target_x)**2 + (ny - target_y)**2)
        others_cells.extend([
            RelCell(0, _phase(ndist, max_dist), 0),
            RelCell(1, _phase(ntarget_dist, max_dist), 0),
            RelCell(2, 0.0, 0),
        ])

    world_cells = [
        RelCell(0, _phase(dist, max_dist), 0),  # same as self dist for click games
        RelCell(1, 0.0, 0),
        RelCell(2, 0.0, 0),
    ]

    return RelState(
        self_word=RelWord(self_cells),
        others_word=RelWord(others_cells),
        world_word=RelWord(world_cells),
        game_id=game_id,
        level=level,
        raw_note=f"node=({active_node_x},{active_node_y}) target=({target_x},{target_y})",
    )


# ── Game-specific adapters ────────────────────────────────────────────────────

def encode_wa30_state(
    player_x: float, player_y: float,
    player_rot: int,
    grabbed_box: Optional[Tuple[float, float]],
    box_positions: List[Tuple[float, float]],
    goal_positions: List[Tuple[float, float]],
    level: int = 0,
) -> RelState:
    """
    wa30 adapter. Box positions are 'others'. Grabbed box is special.

    Others: (box_x, box_y, is_grabbed) for each box.
    Active = grabbed (this box is being moved right now).
    """
    others = []
    for bx, by in box_positions:
        is_grabbed = (grabbed_box is not None and
                      abs(bx - grabbed_box[0]) < 2 and abs(by - grabbed_box[1]) < 2)
        others.append((bx, by, is_grabbed))

    return encode_movement_game(
        player_x=player_x, player_y=player_y,
        player_rot=player_rot,
        carrying=grabbed_box is not None,
        goal_positions=goal_positions,
        entity_positions=others,
        game_id="wa30",
        level=level,
    )


def encode_ls20_state(
    player_x: float, player_y: float,
    player_rot: int,
    goal_positions: List[Tuple[float, float]],
    pushbar_positions: List[Tuple[float, float]],  # active pushbars = obstacles
    level: int = 0,
) -> RelState:
    """
    ls20 adapter. Pushbars are 'others' (active=True since they can block).
    """
    others = [(px, py, True) for px, py in pushbar_positions]
    return encode_movement_game(
        player_x=player_x, player_y=player_y,
        player_rot=player_rot,
        carrying=False,
        goal_positions=goal_positions,
        entity_positions=others,
        game_id="ls20",
        level=level,
    )


def encode_r11l_state(
    active_node_x: float, active_node_y: float,
    target_x: float, target_y: float,
    other_nodes: List[Tuple[float, float]],
    level: int = 0,
) -> RelState:
    """r11l adapter."""
    return encode_click_game(
        active_node_x=active_node_x, active_node_y=active_node_y,
        target_x=target_x, target_y=target_y,
        other_nodes=other_nodes,
        game_id="r11l",
        level=level,
    )


# ── Generic frame-based encoder (no internal state access) ──────────────────

def encode_from_frame(
    frame,           # np.ndarray (64, 64) uint8 or (64, 64, 3)
    win_signature,   # set of (x, y) pixel positions that appear in win frames
    background,      # set of (x, y) pixel positions that never change
    action_count: int,
    budget: int,
    game_id: str = "",
    level: int = 0,
) -> RelState:
    """
    Fallback encoder when internal game state isn't accessible.
    Uses pixel analysis: centroid of changed pixels = 'self', win_signature = 'goal'.

    Less precise than game-specific adapters but works on any game.
    Used for unseen games where we don't have source access.
    """
    import numpy as np
    if hasattr(frame, '__len__') and len(frame.shape) == 3:
        gray = frame.mean(axis=2)
    else:
        gray = frame.astype(float)

    # Non-background pixels = entities
    entity_pixels = [(x, y) for y in range(64) for x in range(64)
                     if (x, y) not in background and gray[y, x] > 10]

    if not entity_pixels:
        # Fully static frame — treat as start state
        self_x, self_y = 32.0, 32.0
    else:
        self_x = float(sum(p[0] for p in entity_pixels)) / len(entity_pixels)
        self_y = float(sum(p[1] for p in entity_pixels)) / len(entity_pixels)

    if win_signature:
        goal_x = float(sum(p[0] for p in win_signature)) / len(win_signature)
        goal_y = float(sum(p[1] for p in win_signature)) / len(win_signature)
    else:
        goal_x, goal_y = 32.0, 32.0

    budget_fraction = action_count / max(1, budget)

    dx = goal_x - self_x
    dy = goal_y - self_y
    dist = math.sqrt(dx*dx + dy*dy)

    self_cells = [
        RelCell(0, _phase(dist, MAX_DIST), 0),
        RelCell(1, _dir_phase(dx, dy), 0),
        RelCell(2, _phase(budget_fraction, 1.0), 0),  # budget used as carry proxy
    ]

    return RelState(
        self_word=RelWord(self_cells),
        others_word=RelWord([]),
        world_word=RelWord([
            RelCell(0, _phase(dist, MAX_DIST), 0),
            RelCell(1, 0.0, 0),
            RelCell(2, 0.0, 0),
        ]),
        game_id=game_id,
        level=level,
        raw_note=f"frame-based entity_centroid=({self_x:.1f},{self_y:.1f})",
    )


# ── Oracle call budget tracker ────────────────────────────────────────────────

class OracleBudget:
    """Enforces max oracle calls per episode."""

    def __init__(self, max_calls: int = ORACLE_CALLS_PER_EPISODE):
        self.max_calls = max_calls
        self.calls_used = 0
        self.call_log: List[str] = []

    def can_call(self) -> bool:
        return self.calls_used < self.max_calls

    def record_call(self, reason: str) -> bool:
        if not self.can_call():
            return False
        self.calls_used += 1
        self.call_log.append(reason)
        return True

    def reset(self):
        self.calls_used = 0
        self.call_log = []

    def remaining(self) -> int:
        return self.max_calls - self.calls_used

    def __repr__(self):
        return f"OracleBudget({self.calls_used}/{self.max_calls} used)"
