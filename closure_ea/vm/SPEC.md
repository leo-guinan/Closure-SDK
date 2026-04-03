# Closure VM — Specification

The VM is the execution core of the Closure computer.

It is the part that steps machine state forward on `S^3` while reading and writing through DNA tables.

The current crate lives in `closure_ea/vm` and compiles as its own Rust crate.

---

## What The VM Is

The VM is a geometric execution core.

Its lowest-level primitive is still the unit quaternion.
Execution means composing quaternions into machine state.
Branching at the base VM level still means reading sigma from the resulting state.
Resonance execution still means using machine state as the key that addresses DNA memory.

But the VM now also has an explicit rotor-cell substrate for higher computation.

That rotor substrate is not a Shannon bit.
It is a local verification orbit on an Euler plane.

The central rule remains:

- DNA owns memory
- the VM owns execution

The VM does not keep its own storage layer.
It runs against caller-provided DNA tables.

---

## Current Module Layout

The current crate is split into:

- `src/primitives.rs`
- `src/program.rs`
- `src/machine.rs`
- `src/hierarchy.rs`
- `src/cell.rs`
- `src/word.rs`
- `src/arithmetic.rs`
- `src/logic.rs`
- `src/control.rs`
- `src/word_memory.rs`
- `src/lib.rs`

### `primitives.rs`
Owns:

- `decompose()`
- `DecomposeResult`
- `StepResult`

### `program.rs`
Owns:

- `Program`
- `Program::compile()`
- `Program::append_inverse()`
- `Program::table_schema()`
- `Program::to_table()`
- `Program::from_table()`

### `machine.rs`
Owns:

- `Machine`
- `execute()`
- `emit()`
- `build_key()`
- `run_sequential()`
- `run_resonance()`
- `run_resonance_weighted()`
- `save()`
- `restore()`
- `Machine::state_table_schema()`

### `hierarchy.rs`
Owns:

- `HierarchicalMachine`
- `ResonanceConfig`
- `ingest()`
- `ingest_with_tables()`

### `cell.rs`
Owns the native rotor cell surface:

- `EulerPlane`
- `PlaneRelation`
- `TwistSheet`
- `VerificationCell`
- `VerificationLandmark`
- local phase composition, twist tracking, and crossing detection

### `word.rs`
Owns ordered collections of rotor cells:

- `VerificationWord`

### `arithmetic.rs`
Owns phase-native arithmetic:

- `VerificationArithmetic`
- `FullAddOutput`
- `FullSubtractOutput`
- `WordSubtractionResult`

### `logic.rs`
Owns rotor-native logic derived from phase arithmetic and symmetry:

- `VerificationLogic`
- `LogicObservation`

### `control.rs`
Owns the blocked control placeholder layer:

- `VerificationControl`
- `ComparisonObservation`
- `BranchObservation`

### `word_memory.rs`
Owns DNA-backed persistence for rotor words:

- `WordMemory`

### `lib.rs`
Owns:

- crate exports
- tests

Current test count:

- `57` Rust tests passing

---

## Machine Model

The base VM machine still has three quaternion registers and a threshold.

```rust
pub struct Machine {
    pub state: [f64; 4],
    pub previous: [f64; 4],
    pub context: [f64; 4],
    pub epsilon: f64,
    pub ip: usize,
    pub cycle_count: usize,
}
```

### Register meaning

- `state` — current running product
- `previous` — state before the last execute
- `context` — accumulated closure history for the current session

These are not decorative registers.
They are the raw material for addressing memory.

A key can be built from:

- `state`
- `state + previous`
- `state + previous + context`

The register set is small, but it is already enough to give the VM single-key, two-key, and three-key resonance addressing.

---

## Rotor Cell Model

The current computational floor above the raw VM is no longer a bit label.

A cell is:

- an Euler plane
- a local phase on that plane
- completed turn count
- derived twist sheet
- coherence width
- local neighbor-coupling state

Conceptually:

```rust
pub struct VerificationCell {
    plane: EulerPlane,
    phase: f64,
    turns: i64,
    sheet: TwistSheet,
    coherence_width: f64,
    coupling: CouplingState,
}
```

This means the primitive is a local verification orbit with topological branch memory, not a static `0/1` bucket.
It is also no longer treated as perfectly sharp or fully isolated.

### Landmarks on the orbit

The important landmarks are:

- `0` phase → identity
- `π` phase → distinction / opposite
- `2π` phase → returned to the same angle on the inverted sheet
- `4π` phase → verified return on the original sheet

These are exposed through:

- `VerificationCell::landmark()`
- `VerificationCell::distinction_crossings_to(...)`
- `VerificationCell::return_crossings_to(...)`
- `VerificationCell::turns()`
- `VerificationCell::sheet()`
- `VerificationCell::coherence_width()`
- `VerificationCell::coupling()`
- `VerificationCell::coupling_to(...)`

### Why this matters

The bit-like behavior is derived from the orbit, but the carrier also remembers whether it returned on the same or inverted branch:

- no distinction contribution = identity landmark
- one distinction contribution = `π` landmark

So binary arithmetic is implemented as phase accumulation on a twisted carrier, not as truth-table lookup over labels.

### Coherence and coupling

The carrier also stores:

- `coherence_width` — how broad the local phase relation is
- `coupling` — local coupling strength and preferred phase bias for neighboring carriers

These are not yet a full coherence field theory.
But they are now first-class state, not hidden assumptions.

---

## Arithmetic Core

The current arithmetic core is phase-native.

For binary arithmetic on one plane:

- `0` contributes `0`
- `1` contributes `π`

For a full adder with inputs `a`, `b`, and `cin`:

- `total_phase = π(a + b + cin)`
- `sum = total_phase mod 2π`
- `carry = floor(total_phase / 2π)`

That is what `VerificationArithmetic::full_add(...)` implements.

### Interpretation

This means:

- `sum` is the local remainder in the current verification cycle
- `carry` is completed full-cycle overflow
- the local cell state also retains whether that overflow changed sheet
- the local cell state preserves aggregated coherence and coupling state

So carry is not bolted on as a separate boolean artifact.
It is native cycle closure.

### Subtraction

`VerificationArithmetic::full_subtract(...)` uses the same geometry in reverse:

- subtract phase contributions
- if the local result goes negative, lift by one full cycle
- the lift is the borrow event

So borrow is also native cycle structure.

---

## Logic Core

The logic core is now derived from the same phase law family.

### `XOR`

`XOR` is local remainder after combining two contributions.

### `AND`

`AND` is completed full-cycle crossing for two contributions.

### `NOT`

`NOT` is symmetry, implemented as a `π` shift on the same plane.

### `OR`

`OR` is derived, currently implemented from `XOR` and `AND` on the same rotor floor.

This means the logic substrate is no longer truth-table-first.
It is arithmetic-first and geometry-first.
Logic observations also retain the richer carrier state that produced them.

---

## Word Storage And DNA Boundary

`WordMemory` persists rotor words through DNA.

The table schema stores, per cell:

- `word_id`
- `cell_index`
- `plane_x`
- `plane_y`
- `plane_z`
- `phase`
- `turns`
- `coherence_width`
- `coupling_strength`
- `coupling_phase_bias`

Important boundary:

- DNA remains a generic table/search/persistence engine
- VM owns rotor-cell semantics

So this storage belongs in `closure_ea/vm`, not in base DNA.

---

## Execution Modes

### 1. Sequential execution

A `Program` is an in-memory list of instruction quaternions.

```rust
let result = machine.run_sequential(&program, max_steps);
```

### 2. Resonance execution

A DNA table acts as program memory.

Each cycle:

1. build a key from machine registers
2. search the caller-provided table
3. read the matched instruction quaternion
4. execute it
5. branch on sigma

The standard entry point is:

```rust
machine.run_resonance(
    table,
    key_width,
    key_col_groups,
    val_cols,
    max_steps,
)
```

Weighted addressing is also available:

```rust
machine.run_resonance_weighted(
    table,
    key_width,
    key_col_groups,
    val_cols,
    weights,
    max_steps,
)
```

---

## Current Status

### Honest floor now in code

The following are now real:

- base quaternion VM execution
- rotor cell with explicit plane + phase + twist/turn memory
- explicit coherence width on each local carrier
- explicit local coupling state on each local carrier
- exact cell geometry and decode
- phase-native add/sub
- rotor-native `XOR`, `AND`, `NOT`, `OR`
- DNA-backed rotor word persistence

### Not yet finished

Still unfinished:

- compare rebuilt on the rotor floor
- branch rebuilt on the same event family
- multiplication/division on the rotor floor
- full control law derived from coherence + closure + coupling instead of placeholders

So the VM now has the right substrate direction, but the arithmetic/control stack is still incomplete above it.
