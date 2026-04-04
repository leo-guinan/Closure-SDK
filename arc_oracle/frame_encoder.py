"""
frame_encoder.py — Entity-aware state encoder for live ARC frames

Converts a 64x64 numpy frame into a RelState using:
  1. Entity cluster analysis (smallest cluster = player proxy)
  2. L3-normalized self_word (dist/dir to goal from pixel coordinates)
  3. Others_word from remaining entity clusters

This produces oracle-queryable states that match DNA entries stored
from known game positions.
"""

from __future__ import annotations
import math
from typing import Optional, Set, Tuple

from relational_encoder import RelCell, RelWord, RelState
from normalized_encoder import (
    L3Constants, KNOWN_L3, encode_normalized, _phase, GRID_W, GRID_H
)


def encode_from_live_frame(
    frame,                          # np.ndarray (64,64) uint8/int8
    game_id: str,
    level: int = 0,
    l3: Optional[L3Constants] = None,
    win_signature: Optional[Set[Tuple[int, int]]] = None,
    background: Optional[Set[Tuple[int, int]]] = None,
    action_count: int = 0,
    budget: int = 200,
) -> RelState:
    """
    Encode a live ARC frame into a RelState for oracle querying.

    Entity detection:
    - All unique pixel values are clustered by value
    - The smallest non-background entity cluster is used as the player proxy
    - Remaining small entities are treated as 'others'
    - Large clusters (>200 pixels) are treated as background/floor/walls

    L3 normalization:
    - Uses KNOWN_L3 constants for the game (goal centroid, max_dist)
    - Falls back to frame-center goal if game unknown
    """
    import numpy as np

    l3 = l3 or KNOWN_L3.get(game_id)
    if l3 is None:
        l3 = L3Constants(game_id=game_id)

    # Cluster all pixel values
    bg = background or set()
    clusters = {}
    for v in np.unique(frame):
        v = int(v)
        rows, cols = np.where(frame == v)
        if len(rows) == 0:
            continue
        px_set = set(zip(cols.tolist(), rows.tolist()))
        # Skip if all pixels are background
        if bg and px_set.issubset(bg):
            continue
        clusters[v] = {
            'count': len(rows),
            'cx': float(np.mean(cols)),
            'cy': float(np.mean(rows)),
            'pixels': px_set,
        }

    # Separate entities from background-scale clusters
    # Background = large clusters (floor, walls). Entities = small clusters.
    BACKGROUND_THRESHOLD = 200  # pixels — anything larger is background/floor
    entities = {v: c for v, c in clusters.items()
                if c['count'] < BACKGROUND_THRESHOLD}
    
    if not entities:
        # All clusters are background — encode as unknown state
        return _unknown_state(l3, game_id, level, action_count, budget)

    # Player proxy = smallest entity cluster (excluding explicit background pixels)
    # For most ARC games the player is the smallest distinct sprite
    entity_vals = sorted(entities.keys(), key=lambda v: entities[v]['count'])
    player_v = entity_vals[0]
    player_cx = entities[player_v]['cx']
    player_cy = entities[player_v]['cy']

    # Others = remaining entity clusters (boxes, NPCs, routing nodes)
    others_data = []
    for v in entity_vals[1:]:
        e = entities[v]
        # active=True if count < 50 (small enough to be a moveable entity)
        others_data.append((e['cx'], e['cy'], e['count'] < 50))

    # Win signature centroid as goal override if provided
    if win_signature:
        gx = float(sum(p[0] for p in win_signature)) / len(win_signature)
        gy = float(sum(p[1] for p in win_signature)) / len(win_signature)
        l3 = L3Constants(
            game_id=game_id,
            goal_centroid=(gx, gy),
            max_dist_to_goal=l3.max_dist_to_goal,
            budget=budget,
            source='frame',
        )

    # Budget fraction as carry proxy for unknown games
    budget_fraction = action_count / max(1, budget)
    carry = budget_fraction > 0.3  # rough proxy: "mid-task" = carrying something

    return encode_normalized(
        self_x=player_cx,
        self_y=player_cy,
        carrying=carry,
        others=others_data,
        l3=l3,
        level=level,
        raw_note=f"frame pixel=({player_cx:.0f},{player_cy:.0f}) val={player_v} count={entities[player_v]['count']}",
    )


def _unknown_state(l3: L3Constants, game_id: str, level: int,
                   action_count: int, budget: int) -> RelState:
    """Fallback: all-identity state when frame gives no entity information."""
    return RelState(
        self_word=RelWord([
            RelCell(0, math.pi, 0),  # dist = max (unknown = far)
            RelCell(1, 0.0, 0),
            RelCell(2, 0.0, 0),
        ]),
        others_word=RelWord([]),
        world_word=RelWord([
            RelCell(0, 0.0, 0),
            RelCell(1, 0.0, 0),
            RelCell(2, 0.0, 0),
        ]),
        game_id=game_id,
        level=level,
        raw_note="unknown frame state",
    )
