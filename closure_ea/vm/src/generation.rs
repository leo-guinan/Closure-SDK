//! Generation — learn token positions on S³, generate sequences by closure.
//!
//! The VM already has everything needed for generation:
//!   - `Machine::execute()` composes quaternions and tracks state C
//!   - `Machine::prediction()` returns C⁻¹ (what would close current state)
//!   - `Program::append_inverse()` guarantees closure on a sequence
//!   - `Machine::run_resonance()` fetches next instruction from DNA by state key
//!
//! What was missing: `Genome::decode()` — the inverse of embed.
//! The kernel knows C⁻¹ (the prediction), but nothing maps that back to a token.
//!
//! This module adds:
//!   - `Genome`      — token positions on S³, backed by a DNA Table
//!   - `learn()`     — ingest sequences with guaranteed closure, teach from votes
//!   - `generate()`  — emit tokens by iterating prediction → decode → ingest
//!
//! # Relationship to Machine::run_resonance()
//!
//! `run_resonance()` IS generation at the program level:
//!   state → FETCH from DNA by content-key → execute → loop to closure.
//!
//! `generate()` does the same thing at the token/genome level:
//!   state → decode(prediction) → nearest token → ingest → loop to closure.
//!
//! The difference: `run_resonance()` fetches exact program instructions.
//! `generate()` fetches the nearest learned position — generalisation, not recall.
//!
//! # Root cause of broken generation (training fix)
//!
//! Hash-embedded tokens rarely compose to closure (σ stays > ε).
//! Zero closures → noisy votes → genome never converges → generation loops.
//! Fix: call `Program::append_inverse()` on every training sequence.
//! This guarantees closure fires, producing clean vote signals every epoch.
//!
//! # Schema
//!
//! A Genome DNA table has this column layout (8 columns):
//!
//! ```text
//! col 0-3:  token_q (w, x, y, z)  — learned position on S³  [key]
//! col 4-7:  token_q (w, x, y, z)  — same quaternion          [value, for retrieval]
//! ```
//!
//! The key and value are the same quaternion. The table is queried by position
//! (nearest-neighbour resonance) to find the closest stored token.
//! Token strings are stored out-of-band in a `Vec<String>` parallel to row indices.

use std::collections::HashMap;
use std::path::Path;
use closure_rs::embed::bytes_to_sphere4;
use closure_rs::groups::sphere::{sphere_inverse as inverse, sphere_sigma as sigma, IDENTITY};
use closure_rs::table::{ColumnDef, ColumnType, Table};
use crate::machine::Machine;
use crate::program::Program;
use crate::primitives::StepResult;

// ── Column indices for Genome DNA table ─────────────────────────────────────

const KEY_COLS: [usize; 4] = [0, 1, 2, 3];
const VAL_COLS: [usize; 4] = [4, 5, 6, 7];

// ── Genome ───────────────────────────────────────────────────────────────────

/// Learned token positions on S³, backed by a Closure DNA table.
///
/// Each token maps to a unit quaternion. Positions are learned from
/// kernel vote signals via SLERP nudges. The DNA table enables
/// content-addressed lookup (decode) using `search_composite`.
///
/// # Training
///
/// Call `learn()` to populate. Every training sequence is extended with
/// `Program::append_inverse()` to guarantee closure fires.
///
/// # Inference
///
/// Call `decode()` to find the token nearest to a target quaternion.
/// This is the inverse of `embed()`: S³ → token.
pub struct Genome {
    /// The DNA table backing this genome.
    /// Schema: [key_w, key_x, key_y, key_z, val_w, val_x, val_y, val_z]
    table: Table,
    /// Token strings, parallel to row indices in the table.
    tokens: Vec<String>,
    /// Map from token string to row index in the table.
    token_index: HashMap<String, usize>,
    /// SLERP step size for position updates.
    damping: f64,
    /// Closure threshold (must match the training Machine epsilon).
    pub epsilon: f64,
}

impl Genome {
    /// Canonical 8-column schema for a genome table.
    /// key_wxyz = learned position (query key).
    /// val_wxyz = same position (returned on decode for retrieval).
    pub fn schema() -> Vec<ColumnDef> {
        let mk = |name: &str| ColumnDef {
            name: name.into(),
            col_type: ColumnType::F64,
            indexed: false,
            not_null: true,
            unique: false,
        };
        vec![
            mk("key_w"), mk("key_x"), mk("key_y"), mk("key_z"),
            mk("val_w"), mk("val_x"), mk("val_y"), mk("val_z"),
        ]
    }

    /// Create a new empty genome backed by a DNA table at `dir`.
    pub fn create(dir: &Path, damping: f64, epsilon: f64) -> std::io::Result<Self> {
        let table = Table::create(dir, Self::schema())?;
        Ok(Self {
            table,
            tokens: Vec::new(),
            token_index: HashMap::new(),
            damping,
            epsilon,
        })
    }

    /// Open an existing genome from `dir`.
    ///
    /// The token list must be provided externally (e.g. from a sidecar JSON file)
    /// because DNA tables store quaternions, not strings.
    pub fn open(dir: &Path, tokens: Vec<String>, damping: f64, epsilon: f64) -> std::io::Result<Self> {
        let table = Table::open(dir)?;
        let token_index = tokens.iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i))
            .collect();
        Ok(Self { table, tokens, token_index, damping, epsilon })
    }

    /// Number of tokens in the genome.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// True if no tokens have been learned yet.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Get the current S³ position for a token.
    ///
    /// If the token is not in the genome, it is initialised via `embed_bytes`
    /// (SHA-256 → S³) and inserted into the table.
    pub fn get(&mut self, token: &str) -> std::io::Result<[f64; 4]> {
        if let Some(&row) = self.token_index.get(token) {
            let q = self.read_row(row)?;
            return Ok(q);
        }
        // Unseen token: embed deterministically and insert
        let q = bytes_to_sphere4(token.as_bytes(), true);
        self.insert_token(token, q)?;
        Ok(q)
    }

    /// Nudge a token's position toward `vote` via SLERP.
    ///
    /// `vote` is the kernel's `C_before⁻¹` — the precision-weighted
    /// prediction error. This is the belief update step in FEP terms.
    ///
    /// Does nothing if the token is not yet in the genome.
    pub fn teach(&mut self, token: &str, vote: &[f64; 4]) -> std::io::Result<()> {
        if let Some(&row) = self.token_index.get(token) {
            let current = self.read_row(row)?;
            let updated = slerp(&current, vote, self.damping);
            self.write_row(row, updated)?;
        }
        Ok(())
    }

    /// Find the token whose S³ position is nearest to `target_q`.
    ///
    /// This is the generation primitive — the inverse of embed:
    ///   embed:  token  → S³  (deterministic)
    ///   decode: S³     → token  (nearest neighbour in genome)
    ///
    /// Uses `Table::search_composite` for content-addressed resonance lookup.
    ///
    /// Returns `None` if the genome is empty.
    pub fn decode(&mut self, target_q: &[f64; 4]) -> std::io::Result<Option<(String, f64)>> {
        if self.tokens.is_empty() {
            return Ok(None);
        }

        self.table.build_genome()?;

        let hits = self.table.search_composite(
            &[(&KEY_COLS, *target_q)],
            1,
        )?;

        if hits.is_empty() {
            return Ok(None);
        }

        let row = hits[0].index;
        let drift = hits[0].drift;

        if row >= self.tokens.len() {
            return Ok(None);
        }

        Ok(Some((self.tokens[row].clone(), drift)))
    }

    /// Token list (in row order).
    pub fn vocab(&self) -> &[String] {
        &self.tokens
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    fn read_row(&mut self, row: usize) -> std::io::Result<[f64; 4]> {
        Ok([
            self.table.get_field_f64(row, VAL_COLS[0])?,
            self.table.get_field_f64(row, VAL_COLS[1])?,
            self.table.get_field_f64(row, VAL_COLS[2])?,
            self.table.get_field_f64(row, VAL_COLS[3])?,
        ])
    }

    fn write_row(&mut self, row: usize, q: [f64; 4]) -> std::io::Result<()> {
        use closure_rs::table::ColumnValue as CV;
        let values = [
            CV::F64(q[0]), CV::F64(q[1]), CV::F64(q[2]), CV::F64(q[3]),
            CV::F64(q[0]), CV::F64(q[1]), CV::F64(q[2]), CV::F64(q[3]),
        ];
        self.table.update(row, &values)
    }

    fn insert_token(&mut self, token: &str, q: [f64; 4]) -> std::io::Result<()> {
        use closure_rs::table::ColumnValue as CV;
        let row = self.table.insert(&[
            CV::F64(q[0]), CV::F64(q[1]), CV::F64(q[2]), CV::F64(q[3]),
            CV::F64(q[0]), CV::F64(q[1]), CV::F64(q[2]), CV::F64(q[3]),
        ])?;
        self.token_index.insert(token.to_string(), row);
        self.tokens.push(token.to_string());
        Ok(())
    }
}

// ── SLERP ────────────────────────────────────────────────────────────────────

/// Spherical linear interpolation: one step from `a` toward `b` on S³.
/// `t` is the step size (0 = stay at a, 1 = jump to b).
fn slerp(a: &[f64; 4], b: &[f64; 4], t: f64) -> [f64; 4] {
    let mut b = *b;
    let mut d: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    if d < 0.0 {
        b = b.map(|x| -x);
        d = -d;
    }
    if d > 0.9999 {
        return normalize(&b);
    }
    let theta = d.acos();
    let s = theta.sin();
    if s < 1e-8 {
        return *a;
    }
    let scale_a = ((1.0 - t) * theta).sin() / s;
    let scale_b = (t * theta).sin() / s;
    let mut r = [0.0f64; 4];
    for i in 0..4 {
        r[i] = scale_a * a[i] + scale_b * b[i];
    }
    normalize(&r)
}

fn normalize(q: &[f64; 4]) -> [f64; 4] {
    let n: f64 = q.iter().map(|x| x * x).sum::<f64>().sqrt();
    if n < 1e-8 { return IDENTITY; }
    [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
}

// ── LearnResult ──────────────────────────────────────────────────────────────

/// Summary of a `learn()` run.
#[derive(Debug, Clone)]
pub struct LearnResult {
    /// Number of closure events fired across all sequences and epochs.
    pub closures: usize,
    /// Total events ingested.
    pub events: usize,
}

// ── learn() ──────────────────────────────────────────────────────────────────

/// Learn token positions from sequences of token strings.
///
/// Each sequence is extended with `Program::append_inverse()` before ingestion.
/// This guarantees closure fires on every sequence every epoch, producing
/// clean vote signals so genome positions converge.
///
/// # Why append_inverse is required
///
/// Hash-embedded tokens rarely compose to closure (σ stays > ε).
/// Without it: 0 closures → noisy votes → no convergence → generation loops.
/// With it: every sequence fires exactly one closure per epoch.
///
/// `Program::append_inverse()` is already implemented in `vm/src/program.rs`.
/// This function calls it on every training sequence before ingestion.
///
/// # Parameters
///
/// - `genome`     — mutable Genome to train
/// - `sequences`  — training sequences (each is a slice of token strings)
/// - `epochs`     — passes over all sequences
pub fn learn(
    genome: &mut Genome,
    sequences: &[Vec<String>],
    epochs: usize,
) -> std::io::Result<LearnResult> {
    let epsilon = genome.epsilon;
    let mut closures = 0usize;
    let mut events = 0usize;

    for _ in 0..epochs {
        for seq in sequences {
            // Build the quaternion program for this sequence
            let mut program = Program::new();
            for token in seq {
                let q = genome.get(token)?;
                program.push(q);
            }

            // append_inverse() guarantees closure fires at the end
            program.append_inverse();

            // Run the program through a Machine, collecting vote signals
            let mut machine = Machine::new(epsilon);
            let instructions = program.as_slice().to_vec();

            // We need to step manually (not run_sequential) to collect votes
            // The vote is C_before^-1 for each step
            machine.reset();
            let mut token_idx = 0;

            for instr in &instructions {
                let c_before = machine.state;
                let vote = inverse(&c_before);

                match machine.execute(instr) {
                    StepResult::Closure(_) => {
                        // Teach the last real token (not the synthetic close)
                        if token_idx < seq.len() {
                            genome.teach(&seq[token_idx], &vote)?;
                        }
                        closures += 1;
                        events += 1;
                        break;
                    }
                    StepResult::Death(_) => {
                        events += 1;
                        break;
                    }
                    StepResult::Continue(_) | StepResult::Halt(_) => {
                        // Teach real tokens only (not the synthetic close instruction)
                        if token_idx < seq.len() {
                            genome.teach(&seq[token_idx], &vote)?;
                        }
                        events += 1;
                        token_idx += 1;
                    }
                }
            }
        }
    }

    Ok(LearnResult { closures, events })
}

// ── GenerateResult ───────────────────────────────────────────────────────────

/// Result of a `generate()` run.
#[derive(Debug, Clone)]
pub struct GenerateResult {
    /// Seed tokens + generated tokens.
    pub tokens: Vec<String>,
    /// σ(C) after each token (including seed).
    pub sigma_trace: Vec<f64>,
    /// Thermodynamic cost: Σ σ(q_i) for each generated step.
    /// The lower bound σ(closure_element) is verifiable without this trace.
    pub sigma_cost: f64,
    /// True if generation terminated at closure (σ < ε).
    pub closed: bool,
}

// ── generate() ───────────────────────────────────────────────────────────────

/// Generate a token sequence from seed tokens.
///
/// The generation loop (FEP action loop):
///   1. Prime Machine with seed tokens
///   2. `prediction` = `machine.state`⁻¹  (what closes current state)
///   3. `genome.decode(prediction)` → nearest learned token
///   4. Ingest that token, update machine state
///   5. Repeat until closure (self-terminating) or `max_steps`
///
/// Termination is intrinsic: the loop stops when σ(C) < ε,
/// meaning the composition has returned to identity — minimum free energy.
///
/// # Relationship to run_resonance()
///
/// `Machine::run_resonance()` does the same thing at the program level:
///   exact program instruction fetch from a DNA table by state key.
///
/// `generate()` does it at the token level:
///   nearest-neighbour genome lookup by prediction quaternion.
///
/// Both terminate at closure. Both use the same Machine registers.
/// The difference is exact recall (run_resonance) vs generalisation (generate).
///
/// # Parameters
///
/// - `genome`      — trained Genome to decode from
/// - `seed_tokens` — initial tokens to prime the Machine with
/// - `max_steps`   — maximum generation steps after seed
pub fn generate(
    genome: &mut Genome,
    seed_tokens: &[String],
    max_steps: usize,
) -> std::io::Result<GenerateResult> {
    let epsilon = genome.epsilon;
    let mut machine = Machine::new(epsilon);
    let mut output = seed_tokens.to_vec();
    let mut sigma_trace = Vec::new();
    let mut sigma_cost = 0.0f64;
    let mut closed = false;

    // Prime with seed
    for token in seed_tokens {
        let q = genome.get(token)?;
        match machine.execute(&q) {
            StepResult::Closure(_) => {
                sigma_trace.push(machine.gap());
                closed = true;
                return Ok(GenerateResult { tokens: output, sigma_trace, sigma_cost, closed });
            }
            _ => {
                sigma_trace.push(machine.gap());
            }
        }
    }

    // Generate
    for _ in 0..max_steps {
        let gap = machine.gap();
        if gap < epsilon {
            closed = true;
            break;
        }

        let prediction = inverse(&machine.state);

        match genome.decode(&prediction)? {
            None => break,
            Some((token, _dist)) => {
                let q = genome.get(&token)?;
                sigma_cost += sigma(&q);

                match machine.execute(&q) {
                    StepResult::Closure(_) => {
                        output.push(token);
                        sigma_trace.push(machine.gap());
                        closed = true;
                        break;
                    }
                    StepResult::Death(_) => {
                        break;
                    }
                    StepResult::Continue(_) | StepResult::Halt(_) => {
                        output.push(token);
                        sigma_trace.push(machine.gap());
                    }
                }
            }
        }
    }

    Ok(GenerateResult { tokens: output, sigma_trace, sigma_cost, closed })
}

// ── Machine::gap() helper ────────────────────────────────────────────────────

/// Convenience: σ(machine.state). The current distance from closure.
impl Machine {
    pub fn gap(&self) -> f64 {
        sigma(&self.state)
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use closure_rs::groups::sphere::sphere_compose as compose;

    fn tmp_dir(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(name);
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    // ── Genome ───────────────────────────────────────────────────────────────

    #[test]
    fn genome_get_unknown_returns_unit() {
        let dir = tmp_dir("gen_get_unit");
        let mut g = Genome::create(&dir, 0.05, 0.15).unwrap();
        let q = g.get("hello").unwrap();
        let n: f64 = q.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((n - 1.0).abs() < 1e-9, "embed must produce unit quaternion");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn genome_get_is_deterministic() {
        let dir = tmp_dir("gen_deterministic");
        let mut g = Genome::create(&dir, 0.05, 0.15).unwrap();
        let q1 = g.get("cat").unwrap();
        let q2 = g.get("cat").unwrap();
        let gap = sigma(&compose(&q1, &inverse(&q2)));
        assert!(gap < 1e-10, "same token must return same position");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn genome_different_tokens_differ() {
        let dir = tmp_dir("gen_differ");
        let mut g = Genome::create(&dir, 0.05, 0.15).unwrap();
        let q1 = g.get("cat").unwrap();
        let q2 = g.get("dog").unwrap();
        let gap = sigma(&compose(&q1, &inverse(&q2)));
        assert!(gap > 0.01, "different tokens must have different positions");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn genome_teach_moves_position() {
        let dir = tmp_dir("gen_teach_moves");
        let mut g = Genome::create(&dir, 0.5, 0.15).unwrap(); // high damping
        let q_before = g.get("word").unwrap();
        let vote = inverse(&q_before); // point in opposite direction
        g.teach("word", &vote).unwrap();
        let q_after = g.get("word").unwrap();
        let moved = sigma(&compose(&q_before, &inverse(&q_after)));
        assert!(moved > 1e-6, "teach must move the position");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn genome_decode_finds_nearest() {
        let dir = tmp_dir("gen_decode_nearest");
        let mut g = Genome::create(&dir, 0.05, 0.15).unwrap();

        // Insert three tokens at known positions
        let near = [0.9f64,  0.1, 0.1, 0.1];
        let mid  = [0.7f64,  0.5, 0.1, 0.1];
        let far  = [0.1f64,  0.9, 0.1, 0.1];

        let near = normalize(&near);
        let mid  = normalize(&mid);
        let far  = normalize(&far);

        // Insert manually via get (will use embed), then overwrite via teach
        g.get("near").unwrap();
        g.get("mid").unwrap();
        g.get("far").unwrap();

        // Force positions by teaching with high damping many times
        for _ in 0..100 {
            g.teach("near", &near).unwrap();
            g.teach("mid",  &mid).unwrap();
            g.teach("far",  &far).unwrap();
        }

        // Query very close to "near"
        let target = normalize(&[0.91f64, 0.09, 0.1, 0.1]);
        let result = g.decode(&target).unwrap();
        assert!(result.is_some(), "decode must return a token");
        let (token, _dist) = result.unwrap();
        assert_eq!(token, "near", "decode should find 'near', got '{token}'");
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── append_inverse via Program ────────────────────────────────────────────

    #[test]
    fn append_inverse_guarantees_closure() {
        let dir = tmp_dir("gen_append_inv");
        let mut g = Genome::create(&dir, 0.05, 0.15).unwrap();

        let tokens = vec!["the".to_string(), "cat".to_string(), "sat".to_string()];
        let mut program = Program::new();
        for t in &tokens {
            program.push(g.get(t).unwrap());
        }
        program.append_inverse();

        let mut machine = Machine::new(0.05);
        let result = machine.run_sequential(&program, 100);
        match result {
            StepResult::Closure(_) => {} // correct
            other => panic!("expected Closure after append_inverse, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── learn ────────────────────────────────────────────────────────────────

    #[test]
    fn learn_fires_closures() {
        let dir = tmp_dir("gen_learn_closures");
        let mut g = Genome::create(&dir, 0.05, 0.15).unwrap();
        let seqs = vec![
            vec!["a".to_string(), "b".to_string()],
            vec!["c".to_string(), "d".to_string(), "e".to_string()],
        ];
        let result = learn(&mut g, &seqs, 3).unwrap();
        // 2 seqs * 3 epochs = 6 closures minimum
        assert!(result.closures >= 6,
            "expected >= 6 closures with append_inverse, got {}", result.closures);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn learn_populates_vocab() {
        let dir = tmp_dir("gen_learn_vocab");
        let mut g = Genome::create(&dir, 0.05, 0.15).unwrap();
        let seqs = vec![
            vec!["foo".to_string(), "bar".to_string()],
            vec!["baz".to_string(), "qux".to_string(), "foo".to_string()],
        ];
        learn(&mut g, &seqs, 3).unwrap();
        for token in ["foo", "bar", "baz", "qux"] {
            assert!(g.vocab().iter().any(|t| t == token),
                "'{token}' missing from genome after learning");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── generate ─────────────────────────────────────────────────────────────

    #[test]
    fn generate_output_starts_with_seed() {
        let dir = tmp_dir("gen_starts_with_seed");
        let mut g = Genome::create(&dir, 0.05, 0.15).unwrap();
        let seqs = vec![
            vec!["the".to_string(), "cat".to_string(), "sat".to_string()],
            vec!["the".to_string(), "dog".to_string(), "ran".to_string()],
        ];
        learn(&mut g, &seqs, 8).unwrap();

        let seed = vec!["the".to_string()];
        let result = generate(&mut g, &seed, 10).unwrap();
        assert_eq!(&result.tokens[..seed.len()], seed.as_slice(),
            "output must start with seed");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn generate_output_longer_than_seed() {
        let dir = tmp_dir("gen_longer_than_seed");
        let mut g = Genome::create(&dir, 0.05, 0.15).unwrap();
        let seqs = vec![
            vec!["the".to_string(), "cat".to_string(), "sat".to_string(), "on".to_string(), "the".to_string(), "mat".to_string()],
            vec!["the".to_string(), "dog".to_string(), "ran".to_string(), "to".to_string(), "the".to_string(), "park".to_string()],
            vec!["a".to_string(),   "fox".to_string(), "sat".to_string(), "on".to_string(), "a".to_string(),   "log".to_string()],
        ];
        learn(&mut g, &seqs, 8).unwrap();

        let seed = vec!["the".to_string(), "cat".to_string()];
        let result = generate(&mut g, &seed, 10).unwrap();
        assert!(result.tokens.len() > seed.len(),
            "generate must add tokens beyond the seed");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn generate_sigma_trace_matches_output_length() {
        let dir = tmp_dir("gen_trace_len");
        let mut g = Genome::create(&dir, 0.05, 0.15).unwrap();
        let seqs = vec![vec!["a".to_string(), "b".to_string(), "c".to_string()]];
        learn(&mut g, &seqs, 5).unwrap();

        let seed = vec!["a".to_string()];
        let result = generate(&mut g, &seed, 5).unwrap();
        assert_eq!(result.sigma_trace.len(), result.tokens.len(),
            "sigma trace length must match output token count");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn generate_sigma_cost_nonnegative() {
        let dir = tmp_dir("gen_cost_nn");
        let mut g = Genome::create(&dir, 0.05, 0.15).unwrap();
        let seqs = vec![vec!["x".to_string(), "y".to_string(), "z".to_string()]];
        learn(&mut g, &seqs, 5).unwrap();
        let result = generate(&mut g, &["x".to_string()], 5).unwrap();
        assert!(result.sigma_cost >= 0.0, "sigma cost must be non-negative");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn generate_respects_max_steps() {
        let dir = tmp_dir("gen_max_steps");
        let mut g = Genome::create(&dir, 0.05, 0.15).unwrap();
        let seqs = vec![vec!["a".to_string(), "b".to_string(), "c".to_string(), "d".to_string()]];
        learn(&mut g, &seqs, 3).unwrap();

        let max_steps = 3usize;
        let seed = vec!["a".to_string()];
        let result = generate(&mut g, &seed, max_steps).unwrap();
        let generated = result.tokens.len() - seed.len();
        assert!(generated <= max_steps,
            "generated {generated} tokens, must not exceed max_steps={max_steps}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn compile_inverse_exact() {
        // Triangle inequality: σ(C) <= Σ σ(q_i)
        // C * C^-1 must equal identity to machine precision
        let dir = tmp_dir("gen_compile_inv");
        let mut g = Genome::create(&dir, 0.05, 0.15).unwrap();
        let tokens = ["x", "y", "z", "w"];
        let mut C = IDENTITY;
        let mut path_cost = 0.0f64;
        for t in &tokens {
            let q = g.get(t).unwrap();
            path_cost += sigma(&q);
            C = compose(&C, &q);
        }
        let net = sigma(&C);
        assert!(net <= path_cost + 1e-10,
            "triangle inequality: net sigma {net:.4} must be <= path cost {path_cost:.4}");

        let verify = compose(&C, &inverse(&C));
        assert!(sigma(&verify) < 1e-10, "C * C^-1 must be identity");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
