"""
Genome -- learned token positions on S3.

The genome is the Python analog of Program + DNA table in the Rust VM.
It stores what the system has learned: for each token, the quaternion
position on S3 that best represents it given the training sequences seen.

Positions are learned from kernel vote signals (the V-step in FEP):
  - vote = C_before^-1: the precision-weighted prediction error
  - teach() nudges the token's position toward the vote via SLERP
  - truth-locked positions (count == -1) are never updated

The key method is decode():
  - Given a target quaternion (the kernel's prediction = C^-1)
  - Returns the token whose learned position is closest on S3
  - This is the inverse of embed(): S3 -> token
  - Without it, generation cannot map predictions back to tokens
"""

import numpy as np
from .s3 import sigma, compose, inverse, embed, geodesic_step, IDENTITY


class Genome:
    """Learned token positions on S3.

    Attributes
    ----------
    positions : dict[str, tuple[np.ndarray, int]]
        token -> (unit quaternion on S3, update count)
        count == -1 means truth-locked: teach() will not overwrite.
    damping : float
        SLERP step size for position updates (0 < damping <= 1).
        Smaller = more conservative learning.
    """

    def __init__(self, damping: float = 0.05):
        self.positions: dict = {}
        self.damping = damping

    def get(self, token: str) -> np.ndarray:
        """Return the current S3 position for a token.
        If unseen, initialise via embed() (deterministic hash)."""
        if token in self.positions:
            return self.positions[token][0].copy()
        pos = embed(token)
        self.positions[token] = (pos, 0)
        return pos.copy()

    def teach(self, token: str, vote: np.ndarray):
        """Update the token's position toward the kernel's vote signal.

        The vote is the kernel's C_before^-1 -- the quaternion that would
        have closed the state before this event was ingested.
        It is the precision-weighted prediction error in FEP terms.

        Truth-locked positions (count == -1) are not updated.
        """
        if token in self.positions:
            pos, count = self.positions[token]
            if count == -1:
                return  # truth-locked
            self.positions[token] = (
                geodesic_step(pos, vote, self.damping),
                count + 1
            )
        else:
            self.positions[token] = (vote.copy(), 1)

    def lock(self, token: str, position: np.ndarray):
        """Set an exact, truth-locked position. teach() will not overwrite it."""
        self.positions[token] = (position.copy(), -1)

    def decode(self, target_q: np.ndarray, exclude_prefix: str = "<close:") -> tuple:
        """Find the token whose S3 position is closest to target_q.

        This is the generation primitive -- the inverse of embed():
          embed():  token  -> S3  (deterministic)
          decode(): S3     -> token  (nearest neighbour in genome)

        target_q is typically the kernel's prediction = C^-1, the quaternion
        that would close the current composition. decode() finds which token
        is closest to that prediction in the learned position space.

        Parameters
        ----------
        target_q : np.ndarray
            Target quaternion to find the nearest token for.
        exclude_prefix : str
            Skip tokens with this prefix (default: synthetic close tokens).

        Returns
        -------
        (token, geodesic_distance) or (None, inf) if genome is empty.
        """
        best_token = None
        best_dist = float('inf')
        for token, (pos, _) in self.positions.items():
            if exclude_prefix and token.startswith(exclude_prefix):
                continue
            d = sigma(compose(inverse(pos), target_q))
            if d < best_dist:
                best_dist = d
                best_token = token
        return best_token, best_dist

    def vocab(self, exclude_prefix: str = "<close:") -> list:
        """Real tokens (excludes synthetic close tokens)."""
        return [t for t in self.positions if not t.startswith(exclude_prefix)]

    def sigma_spread(self, exclude_prefix: str = "<close:") -> tuple:
        """(min, max, mean) of sigma values across real tokens."""
        sigs = [
            sigma(pos)
            for t, (pos, _) in self.positions.items()
            if not t.startswith(exclude_prefix)
        ]
        if not sigs:
            return 0.0, 0.0, 0.0
        return min(sigs), max(sigs), sum(sigs) / len(sigs)

    def __len__(self):
        return len(self.vocab())

    def __repr__(self):
        mn, mx, mean = self.sigma_spread()
        return (f"Genome(tokens={len(self)}, "
                f"sigma=[{mn:.3f}, {mx:.3f}], mean={mean:.3f}, "
                f"damping={self.damping})")
