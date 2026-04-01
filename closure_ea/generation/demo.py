"""
Generation demo for closure_ea.

Run:
  python -m closure_ea.generation.demo

or from the repo root:
  python closure_ea/generation/demo.py

Requires only: numpy
"""

import sys
import os

# Allow running directly from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from closure_ea.generation.s3 import compose, inverse, sigma, IDENTITY
from closure_ea.generation.generator import learn, generate


def run():
    print("=" * 62)
    print("CLOSURE EA: GENERATION DEMO")
    print("S3 algebra  |  FEP figure-8  |  decode() closes the loop")
    print("=" * 62)

    sequences = [
        ["the", "cat", "sat", "on", "the", "mat"],
        ["the", "dog", "ran", "to", "the", "park"],
        ["the", "cat", "ran", "on", "the", "mat"],
        ["a",   "fox", "sat", "on", "a",   "log"],
        ["the", "fox", "ran", "to", "the", "den"],
        ["a",   "cat", "sat", "on", "a",   "mat"],
        ["the", "dog", "sat", "on", "the", "floor"],
    ]

    # ── LEARN ──────────────────────────────────────────────────────

    print(f"\nLEARN")
    print(f"  {len(sequences)} sequences · 8 epochs · epsilon=0.15")
    print(f"  append_inverse() on each sequence: guaranteed closure")
    print(f"  Without this: 0 closures. Votes noisy. Genome never converges.")

    genome, closures, total_events = learn(
        sequences, epsilon=0.15, epochs=8, damping=0.05
    )

    vocab = genome.vocab()
    mn, mx, mean = genome.sigma_spread()
    print(f"\n  {len(vocab)} tokens  |  {total_events} events  |  {closures} closures")
    print(f"  sigma: min={mn:.3f}  max={mx:.3f}  mean={mean:.3f}")
    print()
    print(f"  {'token':<10}  {'sigma':>6}  {'bar':<14}  n")
    print(f"  {'-'*10}  {'-'*6}  {'-'*14}  -")
    for token in sorted(vocab):
        pos, count = genome.positions[token]
        s = sigma(pos)
        bar = "█" * int(s * 9)
        print(f"  {token:<10}  {s:>6.4f}  {bar:<14}  {count}")

    # ── GENERATE ───────────────────────────────────────────────────

    print(f"\nGENERATE")
    print(f"  seed -> prime kernel -> prediction = C^-1 -> decode -> ingest -> repeat")
    print(f"  stops at closure (self-terminating)")

    seeds = [["the", "cat"], ["the", "dog"], ["a", "fox"], ["the"]]
    for seed in seeds:
        out, trace, cost = generate(genome, seed, max_steps=12, epsilon=0.15)
        generated = out[len(seed):]
        final_s = trace[-1] if trace else 0.0
        closed = final_s < 0.15
        status = "CLOSED" if closed else f"open  final sigma={final_s:.3f}"
        print(f"\n  seed:      {' '.join(seed)}")
        print(f"  output:    {' '.join(out)}")
        print(f"  generated: {generated if generated else '(already closed)'}")
        print(f"  sigma:     {[f'{s:.3f}' for s in trace]}")
        print(f"  cost Sigma-sigma: {cost:.4f}   [{status}]")

    # ── COMPILE ────────────────────────────────────────────────────

    print(f"\nCOMPILE")
    print(f"  Program.compile() in vm/src/program.rs: N instructions -> 1 quaternion")
    print(f"  Net sigma = verifiable lower bound (O(1), no trace replay)")
    print(f"  Path cost = private. Waste = path - net = computation not in output.")

    example = ["the", "cat", "sat", "on", "the", "mat"]
    C = IDENTITY.copy()
    path_cost = 0.0
    for t in example:
        q = genome.get(t)
        path_cost += sigma(q)
        C = compose(C, q)
    net = sigma(C)
    waste = path_cost - net
    verify = compose(C, inverse(C))

    print(f"\n  sequence: {example}")
    print(f"  closure element: [{C[0]:.4f} {C[1]:.4f} {C[2]:.4f} {C[3]:.4f}]")
    print(f"  net sigma:       {net:.4f}   <- market prices this")
    print(f"  path cost:       {path_cost:.4f}   <- stays private")
    print(f"  waste:           {waste:.4f}   <- path - net")
    print(f"  C * C^-1:        sigma={sigma(verify):.2e}  (machine epsilon)")

    # ── HOW IT CONNECTS TO THE RUST VM ─────────────────────────────

    print(f"""
HOW THIS CONNECTS TO THE RUST VM

  vm/src/machine.rs:  Machine::execute()
    Identical to _Kernel.ingest() in this module.
    state = C,  prediction = inverse(C),  sigma = gap.

  vm/src/program.rs:  Program.append_inverse()
    Identical to append_inverse() in generator.py.
    Same guarantee: composed sequence fires closure.

  vm/src/machine.rs:  Machine::run_resonance()
    IS the generation loop at the Rust level:
      state -> FETCH from DNA table by content key -> execute -> loop.
    This Python module implements the same loop against the Genome dict
    via decode() (nearest-neighbour) instead of DNA table search.

  The missing piece was decode():
    The VM knows C^-1 (prediction) on every step.
    Nothing mapped that quaternion back to a token.
    decode() = nearest-neighbour in genome position space.
    With it: generate() works. Without it: impossible.

  Root cause of "couldn't get generation perfectly":
    Hash embeddings rarely compose to closure (sigma stays > 0.15).
    0 closures -> noisy votes -> genome doesn't converge -> loops.
    Fix: append_inverse() on training sequences. 0 -> 56 closures.
    Walter already wrote Program.append_inverse() in program.rs.
    Apply it during training. Everything else follows.
""")
    print("=" * 62)


if __name__ == "__main__":
    run()
