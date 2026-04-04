"""
Tests for closure_ea.generation

Mirrors the test style in closure_ea/vm/src/lib.rs:
  - One assertion per test
  - Explicit failure messages
  - No mocking: uses the real S3 algebra
"""

import math
import numpy as np
import pytest

from closure_ea.generation.s3 import (
    compose, inverse, sigma, normalize, embed, geodesic_step, IDENTITY
)
from closure_ea.generation.genome import Genome
from closure_ea.generation.generator import (
    _Kernel, append_inverse, learn, generate
)


# ---------------------------------------------------------------------------
# S3 algebra
# ---------------------------------------------------------------------------

def test_compose_with_identity_is_noop():
    q = normalize(np.array([0.5, 0.3, 0.1, 0.7]))
    r = compose(q, IDENTITY)
    assert sigma(compose(r, inverse(q))) < 1e-10


def test_compose_with_inverse_gives_identity():
    q = normalize(np.array([0.2, 0.8, 0.1, 0.3]))
    result = compose(q, inverse(q))
    assert sigma(result) < 1e-10


def test_compose_is_not_commutative():
    a = normalize(np.array([0.5, 0.3, 0.1, 0.0]))
    b = normalize(np.array([0.1, 0.7, 0.2, 0.3]))
    ab = compose(a, b)
    ba = compose(b, a)
    assert sigma(compose(ab, inverse(ba))) > 0.01, "S3 compose must be non-commutative"


def test_sigma_identity_is_zero():
    assert sigma(IDENTITY) < 1e-10


def test_sigma_increases_with_rotation():
    q1 = normalize(np.array([math.cos(0.15), math.sin(0.15), 0, 0]))
    q2 = normalize(np.array([math.cos(0.45), math.sin(0.45), 0, 0]))
    assert sigma(q2) > sigma(q1)


def test_embed_is_deterministic():
    q1 = embed("hello")
    q2 = embed("hello")
    assert sigma(compose(q1, inverse(q2))) < 1e-10


def test_embed_different_tokens_differ():
    q1 = embed("cat")
    q2 = embed("dog")
    assert sigma(compose(q1, inverse(q2))) > 0.01


def test_embed_is_unit():
    q = embed("anything")
    assert abs(np.linalg.norm(q) - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

def test_kernel_starts_at_identity():
    k = _Kernel()
    assert k.gap < 1e-10


def test_kernel_prediction_closes_state():
    """prediction = C^-1. Composing C with C^-1 must give identity."""
    k = _Kernel(epsilon=0.01)
    q = normalize(np.array([0.6, 0.5, 0.4, 0.3]))
    k.ingest(q)
    pred = k.prediction
    result = compose(k.C, pred)
    assert sigma(result) < 1e-10, "prediction must be exact inverse of state"


def test_kernel_fires_closure():
    """append_inverse logic: composing q then q^-1 must fire closure."""
    k = _Kernel(epsilon=0.05)
    q = normalize(np.array([0.7, 0.4, 0.2, 0.1]))
    k.ingest(q)
    result, _ = k.ingest(inverse(q))
    assert result == 'closure', "q followed by q^-1 must fire closure"


def test_kernel_resets_after_closure():
    k = _Kernel(epsilon=0.05)
    q = normalize(np.array([0.7, 0.4, 0.2, 0.1]))
    k.ingest(q)
    k.ingest(inverse(q))
    assert k.gap < 1e-10, "kernel state must reset to identity after closure"


def test_kernel_vote_is_c_before_inverse():
    """vote = C_before^-1. It must compose with C_before to give identity."""
    k = _Kernel(epsilon=0.01)
    q = normalize(np.array([0.5, 0.6, 0.3, 0.2]))
    C_before = k.C.copy()
    _, vote = k.ingest(q)
    result = compose(C_before, vote)
    # vote = C_before^-1, so C_before * vote = identity... wait,
    # vote = inverse(C_before), so compose(vote, C_before) = identity
    result2 = compose(vote, C_before)
    assert sigma(result2) < 1e-10, "vote must be exact inverse of C_before"


# ---------------------------------------------------------------------------
# Genome
# ---------------------------------------------------------------------------

def test_genome_get_unknown_returns_unit():
    g = Genome()
    q = g.get("novel_token")
    assert abs(np.linalg.norm(q) - 1.0) < 1e-10


def test_genome_teach_moves_position():
    g = Genome(damping=0.5)  # high damping to see movement
    q_before = g.get("word").copy()
    vote = inverse(embed("word"))  # arbitrary vote direction
    g.teach("word", vote)
    q_after = g.get("word")
    dist = sigma(compose(q_before, inverse(q_after)))
    assert dist > 1e-6, "teach() must move the token position"


def test_genome_lock_prevents_teach():
    g = Genome()
    locked_pos = embed("locked")
    g.lock("locked", locked_pos)
    vote = inverse(locked_pos)  # point in opposite direction
    g.teach("locked", vote)
    q_after = g.get("locked")
    dist = sigma(compose(locked_pos, inverse(q_after)))
    assert dist < 1e-10, "locked positions must not be moved by teach()"


def test_genome_decode_finds_nearest():
    """decode() must return the token whose position is closest to target."""
    g = Genome()
    # Place three tokens at known positions
    g.lock("near", normalize(np.array([0.9, 0.1, 0.1, 0.1])))
    g.lock("mid",  normalize(np.array([0.7, 0.5, 0.1, 0.1])))
    g.lock("far",  normalize(np.array([0.1, 0.9, 0.1, 0.1])))

    # Query very close to "near"
    target = normalize(np.array([0.91, 0.09, 0.1, 0.1]))
    token, dist = g.decode(target)
    assert token == "near", f"decode should return 'near', got '{token}'"
    assert dist < 0.5


def test_genome_decode_excludes_close_tokens():
    g = Genome()
    g.lock("<close:test_3>", IDENTITY)
    g.lock("real", embed("real"))
    token, _ = g.decode(IDENTITY)
    assert token == "real", "decode must skip synthetic close tokens"


def test_genome_vocab_excludes_close_tokens():
    g = Genome()
    _ = g.get("cat")
    g.lock("<close:cat_1>", IDENTITY)
    assert "cat" in g.vocab()
    assert "<close:cat_1>" not in g.vocab()


# ---------------------------------------------------------------------------
# append_inverse
# ---------------------------------------------------------------------------

def test_append_inverse_guarantees_closure():
    """The extended sequence must fire closure every time."""
    g = Genome()
    seq = ["the", "cat", "sat"]
    closed = append_inverse(seq, g)
    assert len(closed) == len(seq) + 1

    k = _Kernel(epsilon=0.05)
    fired = False
    for token in closed:
        q = g.get(token)
        result, _ = k.ingest(q)
        if result == 'closure':
            fired = True
            break
    assert fired, "append_inverse must produce a sequence that fires closure"


def test_append_inverse_does_not_modify_original():
    g = Genome()
    seq = ["a", "b", "c"]
    original = list(seq)
    _ = append_inverse(seq, g)
    assert seq == original, "append_inverse must not mutate the input list"


def test_append_inverse_close_token_is_locked():
    g = Genome()
    seq = ["x", "y"]
    closed = append_inverse(seq, g)
    close_token = closed[-1]
    assert close_token.startswith("<close:")
    pos, count = g.positions[close_token]
    assert count == -1, "close token must be truth-locked (count == -1)"


# ---------------------------------------------------------------------------
# learn
# ---------------------------------------------------------------------------

def test_learn_fires_closures():
    """With append_inverse, every sequence fires closure every epoch."""
    seqs = [["a", "b"], ["c", "d", "e"]]
    _, closures, _ = learn(seqs, epochs=3)
    # 2 seqs * 3 epochs = 6 closures minimum
    assert closures >= 6, f"expected >= 6 closures, got {closures}"


def test_learn_updates_positions():
    """After learning, positions should differ from raw embed."""
    seqs = [["the", "cat", "sat"], ["the", "dog", "ran"]]
    genome, _, _ = learn(seqs, epochs=5, damping=0.1)

    raw_the = embed("the")
    learned_the = genome.get("the")
    dist = sigma(compose(raw_the, inverse(learned_the)))
    assert dist > 1e-6, "learning must move token positions from their initial embed"


def test_learn_returns_genome_with_all_tokens():
    seqs = [["foo", "bar"], ["baz", "qux", "foo"]]
    genome, _, _ = learn(seqs)
    for token in ["foo", "bar", "baz", "qux"]:
        assert token in genome.positions, f"'{token}' missing from genome after learning"


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

def test_generate_output_starts_with_seed():
    seqs = [["the", "cat", "sat"], ["the", "dog", "ran"]]
    genome, _, _ = learn(seqs, epochs=8)
    seed = ["the"]
    output, _, _ = generate(genome, seed)
    assert output[:len(seed)] == seed, "output must start with seed"


def test_generate_output_longer_than_seed():
    seqs = [
        ["the", "cat", "sat", "on", "the", "mat"],
        ["the", "dog", "ran", "to", "the", "park"],
        ["a",   "fox", "sat", "on", "a",   "log"],
    ]
    genome, _, _ = learn(seqs, epochs=8)
    seed = ["the", "cat"]
    output, _, _ = generate(genome, seed, max_steps=10)
    assert len(output) > len(seed), "generate must add tokens after seed"


def test_generate_sigma_trace_length_matches_output():
    seqs = [["a", "b", "c"], ["d", "e", "f"]]
    genome, _, _ = learn(seqs, epochs=5)
    seed = ["a"]
    output, trace, _ = generate(genome, seed, max_steps=5)
    assert len(trace) == len(output), (
        f"sigma trace length {len(trace)} must match output length {len(output)}"
    )


def test_generate_cost_is_nonnegative():
    seqs = [["one", "two", "three"]]
    genome, _, _ = learn(seqs, epochs=5)
    _, _, cost = generate(genome, ["one"], max_steps=5)
    assert cost >= 0.0, "sigma cost must be non-negative"


def test_generate_terminates_at_max_steps():
    seqs = [["a", "b", "c", "d", "e"]]
    genome, _, _ = learn(seqs, epochs=3)
    max_steps = 4
    output, _, _ = generate(genome, ["a"], max_steps=max_steps)
    generated_count = len(output) - 1  # minus seed
    assert generated_count <= max_steps, (
        f"generate must not exceed max_steps={max_steps}, got {generated_count}"
    )


def test_generate_compile_inverse_exact():
    """C * C^-1 must equal identity to machine precision."""
    seqs = [["x", "y", "z"]]
    genome, _, _ = learn(seqs, epochs=5)
    C = IDENTITY.copy()
    for t in ["x", "y", "z"]:
        C = compose(C, genome.get(t))
    verify = compose(C, inverse(C))
    assert sigma(verify) < 1e-10, "C * C^-1 must be identity (machine epsilon)"


def test_net_sigma_leq_path_cost():
    """Triangle inequality: sigma(C) <= sum(sigma(q_i))."""
    seqs = [["p", "q", "r", "s"]]
    genome, _, _ = learn(seqs, epochs=5)
    tokens = ["p", "q", "r", "s"]
    C = IDENTITY.copy()
    path_cost = 0.0
    for t in tokens:
        q = genome.get(t)
        path_cost += sigma(q)
        C = compose(C, q)
    net = sigma(C)
    assert net <= path_cost + 1e-10, (
        f"triangle inequality violated: net sigma {net:.4f} > path cost {path_cost:.4f}"
    )
