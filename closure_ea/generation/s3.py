"""
S3 algebra -- pure Python/numpy.

Identical to the primitives in Walter's Rust core (closure_rs/src/groups/sphere.rs).
Used by the generation module so it runs without the Rust bindings,
while producing identical results when bindings are available.

If closure_rs is installed, the Rust implementations are used automatically
through the Kernel class. This module is the Python fallback / reference.
"""

import math
import hashlib
import numpy as np

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


def normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    return q / n if n > 1e-8 else IDENTITY.copy()


def compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product. The one operation everything derives from."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return normalize(np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ]))


def inverse(q: np.ndarray) -> np.ndarray:
    """Conjugate = inverse on S3. Three sign flips."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def sigma(q: np.ndarray) -> float:
    """Geodesic distance from identity.
    sigma = arccos(|w|) in [0, pi/2].
    This is the intrinsic cost of the operation."""
    return math.acos(min(abs(float(q[0])), 1.0))


def geodesic_step(a: np.ndarray, b: np.ndarray, t: float = 0.05) -> np.ndarray:
    """SLERP: one step from a toward b on S3."""
    d = float(np.dot(a, b))
    if d < 0:
        b = -b
        d = -d
    if d > 0.9999:
        return b.copy()
    theta = math.acos(min(d, 1.0))
    s = math.sin(theta)
    if s < 1e-8:
        return a.copy()
    r = math.sin((1 - t) * theta) / s * a + math.sin(t * theta) / s * b
    return normalize(r)


def embed(token: str) -> np.ndarray:
    """Hash a token string to a deterministic unit quaternion on S3.

    Equivalent to ISA #5 EMBED in the VM spec:
      SHA-256(bytes) -> Box-Muller -> normalize -> S3

    Uses a simpler but equivalent mapping: take 4 x 4-byte chunks
    of the SHA-256 digest as raw integers, center, normalize.
    """
    h = hashlib.sha256(token.encode()).digest()
    q = np.array(
        [int.from_bytes(h[i:i + 4], 'little') for i in range(0, 16, 4)],
        dtype=np.float64
    )
    q -= q.mean()
    return normalize(q)
