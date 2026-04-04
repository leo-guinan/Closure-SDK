# Closure EA

`closure_ea` is the umbrella computer module.

## Structure

- `closure_ea/dna/` — persistent geometric memory and database layer
- `closure_ea/vm/` — S³ virtual machine and execution layer
- `closure_ea/enkidu_alive/` — self-contained demo that stays active
- `closure_ea/archive/legacy_runtime/` — archived pre-computer EA runtime files

## Shared Rust Core

DNA and VM both use the shared Rust core in `rust/`.
The shared crate exports the low-level algebra plus the DNA table engine.

## Python Surface

- import DNA from `closure_ea.dna`
- run the DNA CLI with `python -m closure_ea.dna`
- import generation from `closure_ea.generation`
- run the generation demo with `python -m closure_ea.generation.demo`

The old Trinity-era EA runtime has been archived so the top-level module now reflects the computer stack directly.

## Generation

`closure_ea/generation/` adds the Python generation layer to the S³ VM.

The VM already has `Machine::run_resonance()` for content-addressed
generation in Rust. This module exposes the same loop in Python against
a learned token genome, and adds the missing `decode()` primitive.

**The missing piece**: `decode(target_q)` — nearest-neighbour lookup
from a quaternion prediction back to a token. Without it, the kernel
knows `C^-1` (what would close the current state) but nothing maps
that back to a token. Generation is impossible without it.

**The training fix**: `append_inverse(tokens, genome)` — Python
equivalent of `Program.append_inverse()` in `vm/src/program.rs`.
Appends the exact inverse of a sequence so training always fires
closure. Without this, hash embeddings rarely close (σ stays > 0.15),
votes are noisy, and the genome never converges.

```python
from closure_ea.generation import Genome, generate, append_inverse
from closure_ea.generation.generator import learn

sequences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "ran", "to", "the", "park"],
]
genome, closures, _ = learn(sequences, epochs=8)
output, sigma_trace, cost = generate(genome, ["the", "cat"])
print(output)   # ['the', 'cat', ...]
print(cost)     # thermodynamic cost of this generation run (Σσ)
```

Run the demo:
```
python -m closure_ea.generation.demo
```

Run the tests:
```
python -m pytest closure_ea/generation/tests/ -v
```
