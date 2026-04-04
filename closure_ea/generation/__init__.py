"""
closure_ea.generation
=====================
Generation primitives for the S3 VM.

The kernel already computes prediction = C^-1 on every step.
This module adds the missing decode path: nearest-neighbor lookup
from a quaternion prediction back to a token in the genome.

Without decode(), generation is impossible:
  - The kernel knows C^-1 (what would close the current state)
  - But nothing maps that quaternion back to a token

With decode(), the full figure-8 closes:
  ingest  -> learn token positions from kernel vote signals
  decode  -> map prediction back to the nearest learned token
  generate -> iterate decode + ingest until closure (self-terminating)

Public API
----------
  Genome       -- learned token positions on S3
  generate     -- run the generation loop from seed tokens
  append_inverse -- guarantee closure on a training sequence (Python
                    equivalent of Program.append_inverse() in vm/src/program.rs)
"""

from .genome import Genome
from .generator import generate, append_inverse

__all__ = ["Genome", "generate", "append_inverse"]
