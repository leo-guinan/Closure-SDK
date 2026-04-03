"""
Tests for the ARC oracle: encode -> store -> retrieve -> execute

Run: python -m pytest arc_oracle/tests/ -v
  or: cd arc_oracle && python tests/test_oracle.py
"""

import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from relational_encoder import (
    RelCell, RelWord, RelState,
    OracleBudget, MATCH_THRESHOLD,
)
from normalized_encoder import (
    encode_wa30_normalized, encode_ls20_normalized, encode_r11l_normalized,
    encode_sc25_normalized, verify_cross_game_transfer,
    WITHIN_GAME_MATCH_THRESHOLD, CROSS_GAME_MATCH_THRESHOLD,
)
from offline_trainer import OfflineTrainer, GameDNA, _WA30_L1_PLAN, _WA30_L1_START
from runtime_oracle import RuntimeOracle, OracleTrigger, OracleResult


class TestRelCell(unittest.TestCase):

    def test_identity_sigma_zero(self):
        a = RelCell(0, 0.0, 0)
        self.assertAlmostEqual(RelCell.sigma_between(a, a), 0.0, places=10)

    def test_sigma_increases_with_phase(self):
        base = RelCell(0, 0.0, 0)
        near = RelCell(0, 0.3, 0)
        far  = RelCell(0, 1.2, 0)
        self.assertLess(RelCell.sigma_between(base, near),
                        RelCell.sigma_between(base, far))

    def test_sigma_symmetric(self):
        a = RelCell(0, 0.7, 0)
        b = RelCell(0, 1.4, 0)
        self.assertAlmostEqual(RelCell.sigma_between(a, b),
                               RelCell.sigma_between(b, a), places=10)

    def test_sigma_bounded(self):
        # sigma is in [0, π/2] for S³
        a = RelCell(0, 0.0, 0)
        b = RelCell(0, math.pi, 0)
        s = RelCell.sigma_between(a, b)
        self.assertGreaterEqual(s, 0.0)
        self.assertLessEqual(s, math.pi / 2 + 1e-9)

    def test_geometry_unit_quaternion(self):
        c = RelCell(0, 0.8, 0)
        q = c.geometry()
        norm = math.sqrt(sum(x*x for x in q))
        self.assertAlmostEqual(norm, 1.0, places=10)

    def test_same_phase_different_axis_nonzero_sigma(self):
        a = RelCell(0, 0.8, 0)  # i-axis
        b = RelCell(1, 0.8, 0)  # j-axis
        s = RelCell.sigma_between(a, b)
        self.assertGreater(s, 0.01)


class TestRelWord(unittest.TestCase):

    def test_empty_word_sigma_zero(self):
        w = RelWord([])
        self.assertAlmostEqual(w.word_sigma(RelWord([])), 0.0)

    def test_identical_words_sigma_zero(self):
        cells = [RelCell(0, 0.5, 0), RelCell(1, 1.0, 0), RelCell(2, 0.0, 0)]
        w = RelWord(cells)
        self.assertAlmostEqual(w.word_sigma(w), 0.0, places=10)

    def test_word_sigma_increases_with_distance(self):
        w_near_goal = RelWord([RelCell(0, 0.1, 0), RelCell(1, 0.5, 0), RelCell(2, 0.0, 0)])
        w_far_goal  = RelWord([RelCell(0, 1.4, 0), RelCell(1, 1.0, 0), RelCell(2, 0.0, 0)])
        w_identity  = RelWord([RelCell(0, 0.0, 0), RelCell(1, 0.0, 0), RelCell(2, 0.0, 0)])
        self.assertLess(w_near_goal.word_sigma(w_identity),
                        w_far_goal.word_sigma(w_identity))

    def test_flat_roundtrip(self):
        cells = [RelCell(0, 1.23, 2), RelCell(1, 0.45, 0), RelCell(2, math.pi, 1)]
        w = RelWord(cells)
        flat = w.to_flat()
        w2 = RelWord.from_flat(flat)
        for c1, c2 in zip(w.cells, w2.cells):
            self.assertEqual(c1.plane_axis, c2.plane_axis)
            self.assertAlmostEqual(c1.phase, c2.phase, places=10)
            self.assertEqual(c1.turns, c2.turns)

    def test_json_roundtrip(self):
        cells = [RelCell(0, 0.7, 0), RelCell(1, 1.2, 1)]
        w = RelWord(cells)
        w2 = RelWord.from_json(w.to_json())
        self.assertAlmostEqual(w.word_sigma(w2), 0.0, places=10)


class TestEncoders(unittest.TestCase):

    def test_wa30_encodes_without_error(self):
        state = encode_wa30_normalized(
            player_x=8, player_y=8, player_rot=0, grabbed_box=None,
            box_positions=[(16, 8), (8, 20)],
            level=1,
        )
        self.assertEqual(state.game_id, "wa30")
        self.assertEqual(len(state.self_word.cells), 3)
        self.assertGreater(len(state.others_word.cells), 0)

    def test_wa30_near_goal_smaller_sigma_than_start(self):
        start = encode_wa30_normalized(8, 8, 0, None, [(16,8),(8,20),(20,20)], level=1)
        near  = encode_wa30_normalized(22, 8, 0, None, [(24,12),(8,8),(20,20)], level=1)
        at_goal = encode_wa30_normalized(24, 8, 0, None, [(24,8),(24,12),(24,16)], level=1)

        sigma_near  = near.self_word.word_sigma(at_goal.self_word)
        sigma_start = start.self_word.word_sigma(at_goal.self_word)
        self.assertLess(sigma_near, sigma_start,
                        f"Near state ({sigma_near:.4f}) should be closer to goal "
                        f"than start ({sigma_start:.4f})")

    def test_carrying_changes_self_word(self):
        not_carrying = encode_wa30_normalized(8, 8, 0, None,    [(16,8)], level=1)
        carrying     = encode_wa30_normalized(8, 8, 0, (16, 8), [(16,8)], level=1)
        sigma = not_carrying.self_word.word_sigma(carrying.self_word)
        self.assertGreater(sigma, 0.1, "Carry state should change self_word")

    def test_r11l_encodes(self):
        state = encode_r11l_normalized(6, 19, [(25, 57)], level=0)
        self.assertEqual(state.game_id, "r11l")
        self.assertEqual(len(state.self_word.cells), 3)

    def test_json_roundtrip(self):
        state = encode_wa30_normalized(8, 8, 0, None, [(16,8)], level=1)
        state2 = RelState.from_json(state.to_json())
        self.assertAlmostEqual(state.full_sigma(state2), 0.0, places=10)

    def test_primary_sigma_less_than_full_sigma_for_different_others(self):
        state_a = encode_wa30_normalized(8, 8, 0, None, [(16,8)], level=1)
        state_b = encode_wa30_normalized(8, 8, 0, None, [(40,40),(20,20)], level=1)
        primary = state_a.primary_sigma(state_b)
        full    = state_a.full_sigma(state_b)
        self.assertLessEqual(primary, full)


class TestOfflineTrainer(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.trainer = OfflineTrainer(dna_dir=self.tmpdir)

    def test_train_wa30_seeds_one_entry(self):
        added = self.trainer._train_wa30()
        self.assertEqual(added, 1)
        dna = GameDNA("wa30", self.tmpdir)
        self.assertEqual(len(dna), 1)

    def test_train_ls20_seeds_four_entries(self):
        added = self.trainer._train_ls20()
        self.assertEqual(added, 4)
        dna = GameDNA("ls20", self.tmpdir)
        self.assertEqual(len(dna), 4)

    def test_train_r11l_seeds_one_entry(self):
        added = self.trainer._train_r11l()
        self.assertEqual(added, 1)
        dna = GameDNA("r11l", self.tmpdir)
        self.assertEqual(len(dna), 1)

    def test_train_all_returns_dict(self):
        result = self.trainer.train_all()
        self.assertIn("wa30", result)
        self.assertIn("ls20", result)
        self.assertIn("r11l", result)

    def test_no_retrain_without_force(self):
        self.trainer._train_wa30()
        added = self.trainer._train_wa30(force=False)
        self.assertEqual(added, 0)

    def test_force_retrains(self):
        self.trainer._train_wa30()
        added = self.trainer._train_wa30(force=True)
        self.assertEqual(added, 1)

    def test_stored_plan_matches_confirmed(self):
        self.trainer._train_wa30()
        dna = GameDNA("wa30", self.tmpdir)
        entry = dna.entries[0]
        self.assertEqual(entry.actions, _WA30_L1_PLAN)
        self.assertEqual(entry.source, "confirmed_plan")

    def test_persists_across_reload(self):
        self.trainer._train_wa30()
        dna2 = GameDNA("wa30", self.tmpdir)
        self.assertEqual(len(dna2), 1)
        self.assertEqual(dna2.entries[0].actions, _WA30_L1_PLAN)


class TestDNAQuery(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        trainer = OfflineTrainer(dna_dir=self.tmpdir)
        trainer.train_all()
        self.dna = GameDNA("wa30", self.tmpdir)

    def test_exact_start_state_hits(self):
        """Encoding the exact start state should hit the stored plan."""
        s = _WA30_L1_START
        query = encode_wa30_normalized(
            player_x=s["player_x"], player_y=s["player_y"],
            player_rot=s["player_rot"], grabbed_box=s["grabbed_box"],
            box_positions=s["box_positions"], level=1,
        )
        result = self.dna.query(query, level=1, threshold=0.5, use_full_sigma=True)
        self.assertIsNotNone(result, "Exact start state should hit DNA")
        entry, sigma = result
        self.assertEqual(entry.actions, _WA30_L1_PLAN)
        self.assertLess(sigma, 0.5)

    def test_near_start_state_hits(self):
        """State close to start should still hit."""
        query = encode_wa30_normalized(
            player_x=12, player_y=8, player_rot=90, grabbed_box=None,
            box_positions=[(16,8),(8,20),(20,20)], level=1,
        )
        result = self.dna.query(query, level=1, threshold=1.0, use_full_sigma=True)
        self.assertIsNotNone(result, "Near-start state should hit with loose threshold")

    def test_different_level_misses_with_level_filter(self):
        """Query for level 2 should miss wa30 (only L1 stored)."""
        s = _WA30_L1_START
        query = encode_wa30_normalized(
            player_x=s["player_x"], player_y=s["player_y"],
            player_rot=s["player_rot"], grabbed_box=s["grabbed_box"],
            box_positions=s["box_positions"], level=2,
        )
        result = self.dna.query(query, level=2, threshold=0.5, use_full_sigma=True)
        self.assertIsNone(result)

    def test_primary_key_query_looser(self):
        """Primary key query (self→world only) should be more tolerant."""
        query = encode_wa30_normalized(
            player_x=8, player_y=8, player_rot=0, grabbed_box=None,
            box_positions=[],  # no boxes — others word empty
            level=1,
        )
        result_full    = self.dna.query(query, threshold=0.3, use_full_sigma=True)
        result_primary = self.dna.query(query, threshold=0.3, use_full_sigma=False)
        # Primary (no others) may hit when full (with empty others) misses.
        # Both calls must complete without exception — that's the test.
        # If full hits, primary must also hit (primary is a subset of information).
        if result_full is not None:
            self.assertIsNotNone(result_primary,
                "If full-sigma query hits, primary-sigma query must also hit")


class TestRuntimeOracle(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        trainer = OfflineTrainer(dna_dir=self.tmpdir)
        trainer.train_all()
        self.oracle = RuntimeOracle("wa30", dna_dir=self.tmpdir, max_calls=3)
        self.oracle.new_episode(0, level=1)

    def test_episode_start_trigger_hits(self):
        """Oracle at episode start with exact state should return the plan."""
        s = _WA30_L1_START
        state = encode_wa30_normalized(
            player_x=s["player_x"], player_y=s["player_y"],
            player_rot=s["player_rot"], grabbed_box=s["grabbed_box"],
            box_positions=s["box_positions"], level=1,
        )
        result = self.oracle.consult(state, OracleTrigger.EPISODE_START,
                                     action_count=0, budget=70)
        self.assertTrue(result.hit, f"Oracle should hit at episode start. σ={result.sigma:.4f}")
        self.assertEqual(result.actions, _WA30_L1_PLAN)

    def test_budget_exhaustion_returns_miss(self):
        """After 3 calls, oracle returns budget_exhausted."""
        s = _WA30_L1_START
        state = encode_wa30_normalized(
            player_x=s["player_x"], player_y=s["player_y"],
            player_rot=s["player_rot"], grabbed_box=s["grabbed_box"],
            box_positions=s["box_positions"], level=1,
        )
        for i in range(3):
            self.oracle.consult(state, OracleTrigger.EXPLICIT, i, 70)
        result = self.oracle.consult(state, OracleTrigger.EXPLICIT, 3, 70)
        self.assertFalse(result.hit)
        self.assertEqual(result.source, "budget_exhausted")

    def test_same_trigger_doesnt_fire_twice(self):
        """EPISODE_START trigger should not fire again after first call."""
        s = _WA30_L1_START
        state = encode_wa30_normalized(
            player_x=s["player_x"], player_y=s["player_y"],
            player_rot=s["player_rot"], grabbed_box=s["grabbed_box"],
            box_positions=s["box_positions"], level=1,
        )
        # First call at episode start
        r1 = self.oracle.consult(state, OracleTrigger.EPISODE_START, 0, 70)
        # should_consult returns False for same trigger
        should = self.oracle.should_consult(OracleTrigger.EPISODE_START, 0, 70)
        self.assertFalse(should, "EPISODE_START should not fire twice")

    def test_new_episode_resets_budget(self):
        """new_episode resets budget and trigger tracking."""
        s = _WA30_L1_START
        state = encode_wa30_normalized(
            player_x=s["player_x"], player_y=s["player_y"],
            player_rot=s["player_rot"], grabbed_box=s["grabbed_box"],
            box_positions=s["box_positions"], level=1,
        )
        # Exhaust budget
        for i in range(3):
            self.oracle.consult(state, OracleTrigger.EXPLICIT, i, 70)
        # New episode
        self.oracle.new_episode(1, level=1)
        self.assertEqual(self.oracle.budget.remaining(), 3)
        # Should fire again
        should = self.oracle.should_consult(OracleTrigger.EPISODE_START, 0, 70)
        self.assertTrue(should)

    def test_stats_tracked(self):
        s = _WA30_L1_START
        state = encode_wa30_normalized(
            player_x=s["player_x"], player_y=s["player_y"],
            player_rot=s["player_rot"], grabbed_box=s["grabbed_box"],
            box_positions=s["box_positions"], level=1,
        )
        self.oracle.consult(state, OracleTrigger.EPISODE_START, 0, 70)
        self.oracle.new_episode(1, level=1)  # flush log
        stats = self.oracle.stats()
        self.assertGreater(stats["total_oracle_calls"], 0)
        self.assertIn("hit_rate", stats)


class TestCrossGameTransfer(unittest.TestCase):
    """
    Verify the cross-game transfer claim:
    Near-goal states cluster across games after L3 normalization.

    Key distinction (from normalized_encoder analysis):
    - Near-goal states DO cluster across games: wa30_win ≈ sc25_near_exit ≈ r11l_win
    - Start states do NOT cluster: different games have different start-to-goal ratios
      This is correct behavior, not a bug.
    """

    def test_near_goal_states_cluster_across_games(self):
        """The core transfer claim: near-goal in any game looks like near-goal in any other."""
        result = verify_cross_game_transfer(verbose=False)
        self.assertTrue(result["wins_cluster"],
                        "Near-goal states must cluster across games with normalized encoding")

    def test_near_goal_smaller_than_same_game_far(self):
        """Cross-game near-goal sigma must be less than same-game start-vs-win sigma."""
        result = verify_cross_game_transfer(verbose=False)
        baseline = result["baseline"]
        near_goal_sigmas = [v for k, v in result["results"].items()
                            if "both near goal" in k or "both at win" in k]
        for s in near_goal_sigmas:
            self.assertLess(s, baseline,
                f"Near-goal cross-game sigma {s:.4f} should be < baseline {baseline:.4f}")

    def test_carrying_vs_not_carrying_differs(self):
        carrying     = encode_wa30_normalized(8, 8, 0, (16,8), [(16,8)], level=1)
        not_carrying = encode_wa30_normalized(8, 8, 0, None,   [(16,8)], level=1)
        sigma = carrying.primary_sigma(not_carrying)
        self.assertGreater(sigma, 0.05, "Carry state should affect primary sigma")

    def test_primary_sigma_symmetric(self):
        wa30 = encode_wa30_normalized(8, 8, 0, None, [(16,8)], level=1)
        r11l = encode_r11l_normalized(6, 19, [(25,57)], level=0)
        self.assertAlmostEqual(wa30.primary_sigma(r11l),
                               r11l.primary_sigma(wa30), places=10)

    def test_sc25_near_exit_clusters_with_wa30_win(self):
        """wa30_win ≈ sc25_near_exit — the original claim."""
        wa30_win = encode_wa30_normalized(24, 8, 270, None, [(24,8),(24,12),(24,16)], level=1)
        sc25_exit = encode_sc25_normalized(56, 14, False, level=1)
        same_game_far = encode_wa30_normalized(8, 8, 0, None, [(16,8),(8,20),(20,20)], level=1)

        cross_sigma = wa30_win.primary_sigma(sc25_exit)
        baseline = wa30_win.primary_sigma(same_game_far)
        self.assertLess(cross_sigma, baseline,
            f"wa30_win vs sc25_near_exit ({cross_sigma:.4f}) should be < "
            f"wa30 start-vs-win baseline ({baseline:.4f})")


# ── Integration: wa30 L1 full round-trip ─────────────────────────────────────

class TestWa30L1RoundTrip(unittest.TestCase):
    """
    The key integration test:
    1. Seed wa30 L1 plan into DNA
    2. Encode the actual start state
    3. Query DNA
    4. Verify the exact plan is returned
    5. Verify the plan would win (action count check)
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        trainer = OfflineTrainer(dna_dir=self.tmpdir)
        trainer.train_all()
        self.oracle = RuntimeOracle("wa30", dna_dir=self.tmpdir)
        self.oracle.new_episode(0, level=1)

    def test_wa30_l1_full_round_trip(self):
        # 1. Encode start state using normalized encoder
        s = _WA30_L1_START
        start_state = encode_wa30_normalized(
            player_x=s["player_x"], player_y=s["player_y"],
            player_rot=s["player_rot"], grabbed_box=s["grabbed_box"],
            box_positions=s["box_positions"], level=1,
        )

        # 2. Query via oracle
        result = self.oracle.consult(
            start_state,
            OracleTrigger.EPISODE_START,
            action_count=0,
            budget=70,
        )

        # 3. Verify hit
        self.assertTrue(result.hit,
                        f"wa30 L1 should hit DNA. σ={result.sigma:.4f} "
                        f"threshold={self.oracle.threshold}")
        self.assertEqual(result.source, "confirmed_plan")

        # 4. Verify exact plan
        self.assertEqual(result.actions, _WA30_L1_PLAN,
                         "Returned plan should be the confirmed 26-action wa30 L1 plan")

        # 5. Verify plan length is within budget
        self.assertLessEqual(len(result.actions), 70,
                             f"Plan {len(result.actions)} actions must be within budget 70")

        print(f"\n[ROUND TRIP PASS] wa30 L1: σ={result.sigma:.4f} "
              f"plan={result.actions[:5]}... ({len(result.actions)} actions)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
