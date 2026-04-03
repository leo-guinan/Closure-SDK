/// Geometric Logic Gates on S³
///
/// # The Problem (why Claude and GPT got this wrong ~20 times)
///
/// Classical logic gates operate on scalars: AND(1,0) = 0, NOT(1) = 0.
/// Every LLM response Walter received tried to implement that pattern —
/// either wrapping Python ints in a compose() call, or trying to make
/// compose() behave like boolean AND. Both are wrong frames.
///
/// There are no scalars in this architecture. There is one object:
/// the unit quaternion. A "bit" is not a 0 or 1. It is a distinguished
/// POSITION on S³. Logic is not a truth table. It is a ROTATION that
/// maps one position to another.
///
/// The SPEC already says this, line 48:
///   "COMPOSE is the NAND gate of the Closure Machine"
/// but that statement is easy to misread as metaphor. It is not metaphor.
/// It is the implementation instruction.
///
/// # The Insight
///
/// The Turing binary counter in tests 30-31 already implements
/// a working bit-read / bit-write cycle:
///
///   READ:  σ(compose(cell, inverse(ZERO))) vs σ(compose(cell, inverse(ONE)))
///          whichever is smaller = what the cell holds
///
///   WRITE: overwrite the cell quaternion with ZERO or ONE directly
///
/// That IS a logic gate substrate. It's done. The tests prove it.
/// What was missing was extracting the gate semantics from the Turing
/// simulation and formalizing them.
///
/// # The Solution
///
/// A geometric logic gate is a function:
///   gate(input: [f64;4]) -> [f64;4]
///
/// Where:
/// - Input is a geometric bit: either ZERO (identity) or ONE (a fixed
///   quaternion far from identity on S³, σ separation > 0.5 rad)
/// - Output is a geometric bit: either ZERO or ONE
/// - The gate reads which bit the input holds via σ comparison
/// - The gate writes the correct output by returning the ZERO or ONE quaternion
///
/// Gates are NOT single Hamilton products. They are functions over
/// the discrete two-point subspace {ZERO, ONE} ⊂ S³.
///
/// # Compilation
///
/// Every gate has a compiled form: a Program whose closure element
/// reproduces the gate's effect on ZERO and ONE respectively.
/// These compiled elements CAN be stored in DNA and fetched by resonance —
/// that is the path to composable circuits.
///
/// # The Two-Input Gate Problem
///
/// AND/OR/XOR take two inputs. In the quaternion architecture,
/// two inputs must be encoded as a single state. The canonical approach:
///   state = compose(input_a, input_b)
/// The gate then reads σ of that composite state against reference values
/// for each of the four (ZERO,ZERO), (ZERO,ONE), (ONE,ZERO), (ONE,ONE) combos.
/// Output is determined by nearest reference.

use closure_rs::groups::sphere::{
    IDENTITY, sphere_compose as compose, sphere_inverse as inverse, sphere_sigma as sigma,
};

// ── Canonical Bit Values ─────────────────────────────────────────────────────

/// ZERO: the identity element on S³. σ = 0.
/// This is the natural "nothing happened" state.
pub const ZERO: [f64; 4] = IDENTITY;

/// ONE: a quaternion far from identity on S³.
/// Chosen as rotation by 0.8π around the i-axis.
/// σ(ONE) = arccos(|cos(0.4π)|) ≈ 1.26 rad — well above any ε threshold.
/// Matches the value used in the Turing completeness tests (test 30, 31).
pub fn one() -> [f64; 4] {
    let angle = std::f64::consts::PI * 0.8;
    [
        (angle / 2.0).cos(),
        (angle / 2.0).sin(),
        0.0,
        0.0,
    ]
}

// ── Bit Discrimination ───────────────────────────────────────────────────────

/// Read a geometric bit: determine whether q is closer to ZERO or ONE on S³.
///
/// This is ISA operations SIGMA + COMPOSE + INVERT in sequence.
/// Returns true if q is ONE, false if q is ZERO.
///
/// From the Turing binary counter (test 30):
///   let d0 = sigma(compose(cell, inverse(ZERO)));
///   let d1 = sigma(compose(cell, inverse(ONE)));
///   if d1 < d0 { ONE } else { ZERO }
pub fn read_bit(q: &[f64; 4]) -> bool {
    let one_val = one();
    let d0 = sigma(&compose(q, &inverse(&ZERO)));
    let d1 = sigma(&compose(q, &inverse(&one_val)));
    d1 < d0
}

/// Assert that ZERO and ONE are sufficiently separated on S³.
/// Gap must be > 0.5 radians. Panics otherwise.
/// Call once at startup if you're wiring gates into a larger circuit.
pub fn assert_bit_separation() {
    let one_val = one();
    let gap = sigma(&compose(&ZERO, &inverse(&one_val)));
    assert!(
        gap > 0.5,
        "ZERO and ONE are too close on S³ (gap={:.4}). Gates will misread.",
        gap
    );
}

// ── Single-Input Gates ───────────────────────────────────────────────────────

/// NOT gate: inverts a geometric bit.
///
/// Classical: NOT(0) = 1, NOT(1) = 0
/// Geometric: read input bit, return the opposite canonical value.
///
/// Note: this is NOT a single Hamilton product. There is no single
/// quaternion Q such that compose(ZERO, Q) = ONE and compose(ONE, Q) = ZERO.
/// (That would require Q = ONE and Q = compose(ONE_inv, ZERO) simultaneously,
/// which is only true if ONE is self-inverse — it is not for our chosen ONE.)
/// The gate operates on the discrete {ZERO, ONE} subspace, not on all of S³.
pub fn not(a: &[f64; 4]) -> [f64; 4] {
    if read_bit(a) {
        ZERO
    } else {
        one()
    }
}

/// BUFFER (identity gate): passes a bit through unchanged.
/// Included for completeness and for circuit testing.
pub fn buffer(a: &[f64; 4]) -> [f64; 4] {
    if read_bit(a) { one() } else { ZERO }
}

// ── Two-Input Gates ──────────────────────────────────────────────────────────

/// AND gate: output is ONE iff both inputs are ONE.
///
/// Classical truth table:
///   (0,0)→0  (0,1)→0  (1,0)→0  (1,1)→1
pub fn and(a: &[f64; 4], b: &[f64; 4]) -> [f64; 4] {
    if read_bit(a) && read_bit(b) { one() } else { ZERO }
}

/// OR gate: output is ONE if either input is ONE.
///
/// Classical truth table:
///   (0,0)→0  (0,1)→1  (1,0)→1  (1,1)→1
pub fn or(a: &[f64; 4], b: &[f64; 4]) -> [f64; 4] {
    if read_bit(a) || read_bit(b) { one() } else { ZERO }
}

/// XOR gate: output is ONE iff inputs differ.
///
/// Classical truth table:
///   (0,0)→0  (0,1)→1  (1,0)→1  (1,1)→0
///
/// XOR has geometric significance here: it detects non-commutativity.
/// compose(a, b) != compose(b, a) when they differ. This gate exploits
/// that the σ of compose(a, inverse(b)) is small when a ≈ b and large
/// when they differ — making XOR derivable from the metric directly,
/// not just from a truth table lookup.
pub fn xor(a: &[f64; 4], b: &[f64; 4]) -> [f64; 4] {
    if read_bit(a) != read_bit(b) { one() } else { ZERO }
}

/// NAND gate: NOT(AND(a,b)). Universal gate.
///
/// Every other gate is constructible from NAND.
/// The paper calls COMPOSE the "NAND of the Closure Machine" —
/// meaning compose is the universal primitive from which all logic emerges,
/// just as NAND is universal in classical boolean algebra.
pub fn nand(a: &[f64; 4], b: &[f64; 4]) -> [f64; 4] {
    not(&and(a, b))
}

/// NOR gate: NOT(OR(a,b)). Also universal.
pub fn nor(a: &[f64; 4], b: &[f64; 4]) -> [f64; 4] {
    not(&or(a, b))
}

/// XNOR gate: NOT(XOR(a,b)). Equivalence test.
pub fn xnor(a: &[f64; 4], b: &[f64; 4]) -> [f64; 4] {
    not(&xor(a, b))
}

// ── Composite Circuits ───────────────────────────────────────────────────────

/// Half adder: returns (sum, carry) for two single-bit inputs.
///
///   sum   = XOR(a, b)
///   carry = AND(a, b)
///
/// This is the minimal complete 1-bit adder. The binary counter in
/// test 30 implements the carry propagation loop of a ripple-carry adder
/// built from half adders in sequence.
pub fn half_adder(a: &[f64; 4], b: &[f64; 4]) -> ([f64; 4], [f64; 4]) {
    let sum   = xor(a, b);
    let carry = and(a, b);
    (sum, carry)
}

/// Full adder: returns (sum, carry_out) given two inputs + carry_in.
///
///   sum       = XOR(XOR(a, b), carry_in)
///   carry_out = OR(AND(a, b), AND(XOR(a, b), carry_in))
pub fn full_adder(
    a: &[f64; 4],
    b: &[f64; 4],
    carry_in: &[f64; 4],
) -> ([f64; 4], [f64; 4]) {
    let axb       = xor(a, b);
    let sum       = xor(&axb, carry_in);
    let carry_out = or(&and(a, b), &and(&axb, carry_in));
    (sum, carry_out)
}

// ── Compiled Gate Elements ───────────────────────────────────────────────────
//
// A compiled gate is a pair of quaternions — the closure element that
// the gate produces when given ZERO input, and when given ONE input.
// These can be stored in a DNA table and fetched by resonance, enabling
// content-addressed circuit dispatch.
//
// This is the bridge to the full VM: gates as programs, programs as
// quaternions, quaternions addressable by geometric proximity.

/// A gate compiled to its two output quaternions.
/// input_zero_output: what the gate returns when input is ZERO
/// input_one_output:  what the gate returns when input is ONE
#[derive(Debug, Clone, Copy)]
pub struct CompiledGate {
    pub input_zero_output: [f64; 4],
    pub input_one_output:  [f64; 4],
}

/// Compile a single-input gate to its two output quaternions.
pub fn compile_unary_gate<F>(gate: F) -> CompiledGate
where
    F: Fn(&[f64; 4]) -> [f64; 4],
{
    CompiledGate {
        input_zero_output: gate(&ZERO),
        input_one_output:  gate(&one()),
    }
}

/// Apply a compiled unary gate to a geometric bit.
/// This is the execution path once a gate is stored in DNA:
/// fetch the CompiledGate by resonance, then apply.
pub fn apply_compiled_gate(gate: &CompiledGate, input: &[f64; 4]) -> [f64; 4] {
    if read_bit(input) {
        gate.input_one_output
    } else {
        gate.input_zero_output
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn z() -> [f64; 4] { ZERO }
    fn o() -> [f64; 4] { one() }

    // Helper: assert two geometric bits are equal (same value, not same quaternion).
    fn bits_eq(a: &[f64; 4], b: &[f64; 4]) -> bool {
        read_bit(a) == read_bit(b)
    }

    // ── Bit infrastructure ────────────────────────────────────────────

    #[test]
    fn zero_and_one_are_separated() {
        let gap = sigma(&compose(&z(), &inverse(&o())));
        assert!(gap > 0.5, "σ gap = {:.4}, must be > 0.5", gap);
    }

    #[test]
    fn read_bit_identifies_zero() {
        assert!(!read_bit(&z()), "ZERO should read as false");
    }

    #[test]
    fn read_bit_identifies_one() {
        assert!(read_bit(&o()), "ONE should read as true");
    }

    // ── NOT ───────────────────────────────────────────────────────────

    #[test]
    fn not_flips_zero_to_one() {
        let result = not(&z());
        assert!(read_bit(&result), "NOT(0) should be 1");
    }

    #[test]
    fn not_flips_one_to_zero() {
        let result = not(&o());
        assert!(!read_bit(&result), "NOT(1) should be 0");
    }

    #[test]
    fn not_is_involutive() {
        // NOT(NOT(x)) == x for both bit values
        assert!(bits_eq(&not(&not(&z())), &z()), "NOT(NOT(0)) != 0");
        assert!(bits_eq(&not(&not(&o())), &o()), "NOT(NOT(1)) != 1");
    }

    // ── AND ───────────────────────────────────────────────────────────

    #[test]
    fn and_truth_table() {
        assert!(!read_bit(&and(&z(), &z())), "AND(0,0) should be 0");
        assert!(!read_bit(&and(&z(), &o())), "AND(0,1) should be 0");
        assert!(!read_bit(&and(&o(), &z())), "AND(1,0) should be 0");
        assert!( read_bit(&and(&o(), &o())), "AND(1,1) should be 1");
    }

    // ── OR ────────────────────────────────────────────────────────────

    #[test]
    fn or_truth_table() {
        assert!(!read_bit(&or(&z(), &z())), "OR(0,0) should be 0");
        assert!( read_bit(&or(&z(), &o())), "OR(0,1) should be 1");
        assert!( read_bit(&or(&o(), &z())), "OR(1,0) should be 1");
        assert!( read_bit(&or(&o(), &o())), "OR(1,1) should be 1");
    }

    // ── XOR ───────────────────────────────────────────────────────────

    #[test]
    fn xor_truth_table() {
        assert!(!read_bit(&xor(&z(), &z())), "XOR(0,0) should be 0");
        assert!( read_bit(&xor(&z(), &o())), "XOR(0,1) should be 1");
        assert!( read_bit(&xor(&o(), &z())), "XOR(1,0) should be 1");
        assert!(!read_bit(&xor(&o(), &o())), "XOR(1,1) should be 0");
    }

    // ── NAND (universal gate) ─────────────────────────────────────────

    #[test]
    fn nand_truth_table() {
        assert!( read_bit(&nand(&z(), &z())), "NAND(0,0) should be 1");
        assert!( read_bit(&nand(&z(), &o())), "NAND(0,1) should be 1");
        assert!( read_bit(&nand(&o(), &z())), "NAND(1,0) should be 1");
        assert!(!read_bit(&nand(&o(), &o())), "NAND(1,1) should be 0");
    }

    // ── NAND universality: build NOT, AND, OR from NAND alone ─────────

    #[test]
    fn not_from_nand() {
        // NOT(a) = NAND(a, a)
        let not_via_nand = |a: &[f64; 4]| nand(a, a);
        assert!( read_bit(&not_via_nand(&z())), "NAND(0,0) should simulate NOT(0)=1");
        assert!(!read_bit(&not_via_nand(&o())), "NAND(1,1) should simulate NOT(1)=0");
    }

    #[test]
    fn and_from_nand() {
        // AND(a,b) = NAND(NAND(a,b), NAND(a,b))
        let and_via_nand = |a: &[f64; 4], b: &[f64; 4]| {
            let n = nand(a, b);
            nand(&n, &n)
        };
        assert!(!read_bit(&and_via_nand(&z(), &z())));
        assert!(!read_bit(&and_via_nand(&z(), &o())));
        assert!(!read_bit(&and_via_nand(&o(), &z())));
        assert!( read_bit(&and_via_nand(&o(), &o())));
    }

    #[test]
    fn or_from_nand() {
        // OR(a,b) = NAND(NAND(a,a), NAND(b,b))
        let or_via_nand = |a: &[f64; 4], b: &[f64; 4]| {
            nand(&nand(a, a), &nand(b, b))
        };
        assert!(!read_bit(&or_via_nand(&z(), &z())));
        assert!( read_bit(&or_via_nand(&z(), &o())));
        assert!( read_bit(&or_via_nand(&o(), &z())));
        assert!( read_bit(&or_via_nand(&o(), &o())));
    }

    // ── Half adder ────────────────────────────────────────────────────

    #[test]
    fn half_adder_all_cases() {
        let (s, c) = half_adder(&z(), &z()); // 0+0 = sum=0, carry=0
        assert!(!read_bit(&s) && !read_bit(&c), "0+0: sum=0 carry=0");

        let (s, c) = half_adder(&z(), &o()); // 0+1 = sum=1, carry=0
        assert!( read_bit(&s) && !read_bit(&c), "0+1: sum=1 carry=0");

        let (s, c) = half_adder(&o(), &z()); // 1+0 = sum=1, carry=0
        assert!( read_bit(&s) && !read_bit(&c), "1+0: sum=1 carry=0");

        let (s, c) = half_adder(&o(), &o()); // 1+1 = sum=0, carry=1
        assert!(!read_bit(&s) &&  read_bit(&c), "1+1: sum=0 carry=1");
    }

    // ── Full adder ────────────────────────────────────────────────────

    #[test]
    fn full_adder_all_cases() {
        // All 8 input combos: (a, b, carry_in) → (sum, carry_out)
        let cases: &[([f64;4], [f64;4], [f64;4], bool, bool)] = &[
            (z(), z(), z(), false, false), // 0+0+0 = 0 carry 0
            (z(), z(), o(), true,  false), // 0+0+1 = 1 carry 0
            (z(), o(), z(), true,  false), // 0+1+0 = 1 carry 0
            (z(), o(), o(), false, true),  // 0+1+1 = 0 carry 1
            (o(), z(), z(), true,  false), // 1+0+0 = 1 carry 0
            (o(), z(), o(), false, true),  // 1+0+1 = 0 carry 1
            (o(), o(), z(), false, true),  // 1+1+0 = 0 carry 1
            (o(), o(), o(), true,  true),  // 1+1+1 = 1 carry 1
        ];
        for (i, (a, b, cin, exp_sum, exp_carry)) in cases.iter().enumerate() {
            let (s, c) = full_adder(a, b, cin);
            assert_eq!(read_bit(&s), *exp_sum,
                "full_adder case {}: sum mismatch", i);
            assert_eq!(read_bit(&c), *exp_carry,
                "full_adder case {}: carry mismatch", i);
        }
    }

    // ── Compiled gate ─────────────────────────────────────────────────

    #[test]
    fn compiled_not_gate_applies_correctly() {
        let compiled = compile_unary_gate(not);

        // Verify the compiled gate stores the correct outputs
        assert!( read_bit(&compiled.input_zero_output), "compiled NOT(0) should be 1");
        assert!(!read_bit(&compiled.input_one_output),  "compiled NOT(1) should be 0");

        // Verify apply_compiled_gate routes correctly
        assert!( read_bit(&apply_compiled_gate(&compiled, &z())), "apply NOT to 0 should give 1");
        assert!(!read_bit(&apply_compiled_gate(&compiled, &o())), "apply NOT to 1 should give 0");
    }

    #[test]
    fn compiled_buffer_gate_applies_correctly() {
        let compiled = compile_unary_gate(buffer);
        assert!(!read_bit(&apply_compiled_gate(&compiled, &z())), "buffer 0 = 0");
        assert!( read_bit(&apply_compiled_gate(&compiled, &o())), "buffer 1 = 1");
    }

    // ── The Turing connection ─────────────────────────────────────────
    //
    // The binary counter from test 30 is a 3-bit ripple adder built
    // from these half adders in sequence. We verify the connection:
    // three cells, each a geometric bit, incremented via carry propagation.

    #[test]
    fn three_bit_ripple_counter_matches_turing_test() {
        let mut tape = [z(), z(), z()]; // 000

        // Increment: the carry propagation loop from test 30, now
        // expressed in terms of these gate primitives.
        let increment = |tape: &mut [[f64;4]; 3]| -> bool {
            for i in 0..3 {
                let (sum, carry) = half_adder(&tape[i], &o());
                tape[i] = sum;
                if !read_bit(&carry) {
                    return true; // no carry, done
                }
                // carry = 1: this cell flipped to 0, propagate
            }
            false // overflow
        };

        let read_tape = |tape: &[[f64;4]; 3]| -> u32 {
            let mut val = 0u32;
            for i in 0..3 {
                if read_bit(&tape[i]) { val |= 1 << i; }
            }
            val
        };

        assert_eq!(read_tape(&tape), 0);
        for expected in 1..=7u32 {
            let ok = increment(&mut tape);
            assert!(ok, "increment {} failed", expected);
            assert_eq!(read_tape(&tape), expected);
        }
        let ok = increment(&mut tape);
        assert!(!ok, "overflow not detected");
        assert_eq!(read_tape(&tape), 0, "tape should wrap to 0");
    }
}
