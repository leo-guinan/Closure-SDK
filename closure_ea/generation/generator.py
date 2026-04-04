"""
Generator -- the generation loop for the S3 VM.

The kernel already has everything needed for generation:
  - state C on S3, updated by compose on every ingest
  - prediction = C^-1: the quaternion that would close the current state
  - closure: fires when sigma(C) < epsilon

What was missing: decode() -- mapping C^-1 back to a token.

This module provides:

  append_inverse(tokens, genome)
    Python equivalent of Program.append_inverse() in vm/src/program.rs.
    Appends the exact inverse of the compiled sequence so training
    sequences are guaranteed to fire closure. Required for clean learning.

  learn(sequences, ...)
    Ingest sequences with guaranteed closure. Teach genome positions
    from kernel vote signals. Returns a trained Genome.

  generate(genome, seed_tokens, ...)
    Run the generation loop from a seed. Returns (tokens, sigma_trace, cost).
    Self-terminating: stops at closure or max_steps.
    sigma cost is the thermodynamic cost of this generation run.
"""

import numpy as np
from .s3 import compose, inverse, sigma, IDENTITY
from .genome import Genome


# ---------------------------------------------------------------------------
# Kernel (pure Python, same algebra as Rust Machine in vm/src/machine.rs)
# ---------------------------------------------------------------------------

class _Kernel:
    """Internal kernel. Identical to Machine in closure_ea/vm/src/machine.rs.

    state  = C on S3 (accumulator)
    prediction = C^-1 (downward signal: what would close current state)
    ingest(q) -> compose(C, q), check sigma, return (result, vote)
    """

    def __init__(self, epsilon: float = 0.15):
        self.C = IDENTITY.copy()
        self.epsilon = epsilon

    @property
    def gap(self) -> float:
        return sigma(self.C)

    @property
    def prediction(self) -> np.ndarray:
        """C^-1 -- the quaternion that would close this composition.
        This is the top-down signal in FEP / the query for decode()."""
        return inverse(self.C)

    def ingest(self, q: np.ndarray) -> tuple:
        """Compose q into state. Return ('closure'|'open', vote).

        vote = C_before^-1: the precision-weighted prediction error.
        The adapter uses this to teach the genome.
        """
        C_before = self.C.copy()
        self.C = compose(self.C, q)
        vote = inverse(C_before)
        if sigma(self.C) < self.epsilon:
            self.C = IDENTITY.copy()
            return 'closure', vote
        return 'open', vote

    def reset(self):
        self.C = IDENTITY.copy()


# ---------------------------------------------------------------------------
# append_inverse
# ---------------------------------------------------------------------------

def append_inverse(tokens: list, genome: Genome) -> list:
    """Append the exact inverse of the compiled sequence to tokens.

    Guarantees the extended sequence fires closure (sigma -> 0).
    Without this, hash-embedded sequences rarely close naturally,
    producing zero closure events, noisy vote signals, and a genome
    that never converges -- the root cause of broken generation.

    This is the Python equivalent of Program.append_inverse() in
    closure_ea/vm/src/program.rs. Same guarantee, same algebra.

    The synthetic close token is truth-locked in the genome so
    teach() never overwrites its exact position.

    Parameters
    ----------
    tokens : list[str]
        The training sequence to guarantee closure on.
    genome : Genome
        The genome whose positions are used to compile the sequence.

    Returns
    -------
    list[str] : tokens + [close_token]
    """
    C = IDENTITY.copy()
    for t in tokens:
        C = compose(C, genome.get(t))
    close_q = inverse(C)
    close_token = f"<close:{tokens[0]}_{len(tokens)}>"
    genome.lock(close_token, close_q)
    return tokens + [close_token]


# ---------------------------------------------------------------------------
# learn
# ---------------------------------------------------------------------------

def learn(
    sequences: list,
    epsilon: float = 0.15,
    epochs: int = 8,
    damping: float = 0.05,
) -> tuple:
    """Learn token positions from sequences.

    Uses append_inverse() on every sequence to guarantee closure.
    Teaches the genome from kernel vote signals after each ingest.

    Parameters
    ----------
    sequences : list[list[str]]
        Training sequences. Each is a list of token strings.
    epsilon : float
        Closure threshold. sigma < epsilon fires closure.
    epochs : int
        Number of passes over all sequences.
    damping : float
        SLERP step size for genome position updates.

    Returns
    -------
    (genome, n_closures, n_events) : (Genome, int, int)
    """
    genome = Genome(damping=damping)
    kernel = _Kernel(epsilon=epsilon)
    closures = 0
    total_events = 0

    for _ in range(epochs):
        for seq in sequences:
            closed_seq = append_inverse(seq, genome)
            kernel.reset()
            for token in closed_seq:
                q = genome.get(token)
                result, vote = kernel.ingest(q)
                if not token.startswith('<close:'):
                    genome.teach(token, vote)
                total_events += 1
                if result == 'closure':
                    closures += 1

    return genome, closures, total_events


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

def generate(
    genome: Genome,
    seed_tokens: list,
    max_steps: int = 20,
    epsilon: float = 0.15,
) -> tuple:
    """Generate a token sequence from seed tokens.

    The generation loop (FEP action loop):
      1. Prime kernel with seed tokens
      2. prediction = kernel.C^-1  (what closes the current state)
      3. token = genome.decode(prediction)  (nearest learned token)
      4. ingest token, update state
      5. repeat until closure (self-terminating) or max_steps

    Generation stops at closure because the kernel resets to identity
    when sigma(C) < epsilon -- the free energy minimum has been reached.

    Parameters
    ----------
    genome : Genome
        Trained genome with learned token positions.
    seed_tokens : list[str]
        Initial tokens to prime the kernel with.
    max_steps : int
        Maximum generation steps after the seed.
    epsilon : float
        Closure threshold (should match the training epsilon).

    Returns
    -------
    (output_tokens, sigma_trace, sigma_cost) : (list[str], list[float], float)

    output_tokens  : seed + generated tokens
    sigma_trace    : sigma value after each token (including seed)
    sigma_cost     : sum of sigma(q) for each generated step
                     This is the thermodynamic cost of the generation run.
                     sigma(closure_element) is a verifiable lower bound.
    """
    kernel = _Kernel(epsilon=epsilon)
    output = list(seed_tokens)
    sigma_trace = []
    generated_qs = []

    # Prime with seed
    for token in seed_tokens:
        q = genome.get(token)
        result, _ = kernel.ingest(q)
        sigma_trace.append(kernel.gap)
        if result == 'closure':
            return output, sigma_trace, 0.0

    # Generate
    for _ in range(max_steps):
        if kernel.gap < epsilon:
            break

        target = kernel.prediction
        token, _ = genome.decode(target)
        if token is None:
            break

        q = genome.get(token)
        generated_qs.append(q)
        result, _ = kernel.ingest(q)
        output.append(token)
        sigma_trace.append(kernel.gap)

        if result == 'closure':
            break

    total_cost = sum(sigma(q) for q in generated_qs)
    return output, sigma_trace, total_cost
