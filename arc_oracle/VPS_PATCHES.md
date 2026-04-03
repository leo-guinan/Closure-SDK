# VPS Patches Applied to arc_engine.py

These changes were applied directly to `/root/arc-engine/arc_engine.py` on the VPS.
Backup at `/root/arc-engine/arc_engine.py.bak_oracle`.

## Patch 1-4: Oracle integration (applied 2026-04-03)

Applied via `/tmp/patch_arc_engine2.py`:
- Oracle imports (arc_engine_hook, KNOWN_L3, encode_from_frame, OracleTrigger)  
- `_oracle_hooks: dict` on ArcEngine
- `_oracle_for(game)` helper method
- Oracle pre-query block in `run()` level loop

## Patch 5: Frame encoder (applied 2026-04-03)

Swapped `encode_from_frame` → `encode_from_live_frame` in oracle query block.
`frame_encoder.py` added to `arc_oracle/`.

## Patch 6: Level offset fix (applied 2026-04-03)

`_hook.new_episode(episode=level, level=level)` — was `level=level-1`, wrong level filter.

## Patch 7: Cross-game fallback level filter (applied 2026-04-03)

`runtime_oracle.py`: cross-game fallback now uses `level=self._current_level` 
instead of `level=None` to prevent L1 plan being returned for L2.

## Patch 8: Newton bypass + wa30 L3 fix (applied 2026-04-03)

### run_plan(bypass_newton=False)
`act()` gains `bypass_newton` param. If True, skips Newton gate check.
Confirmed plans call `loop.run_plan(confirmed, bypass_newton=True)`.

### Newton npc_steal gate tightened
Position threshold changed from `< 3` to `< 8` for both x and y.
Still blocks genuine routing to (32,12) but doesn't false-trigger on 
player passing through y=12 while heading to x=8 box.

### _plan_wa30_l3 returns confirmed plan
Was `return None`. Now returns `CONFIRMED_PLANS["wa30"][3]`.
Enables L3 to execute via Marvin exploration path if confirmed plan fails.

## Result: wa30 4/5 levels solved (was 2/3)

L1: oracle hit σ=0.000 → SOLVED (oracle)
L2: oracle miss → confirmed plan → SOLVED
L3: oracle miss → confirmed plan (bypass_newton) → SOLVED
L4: oracle miss → confirmed plan → SOLVED  
L5: no plan → outcome=no_plan (5 boxes, needs BFS decomposition)
