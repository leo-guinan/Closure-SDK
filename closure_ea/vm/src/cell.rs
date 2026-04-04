use std::f64::consts::{PI, TAU};

use closure_rs::groups::sphere::{sphere_compose as compose, IDENTITY};

const CELL_EPSILON: f64 = 1e-9;
const DEFAULT_COHERENCE_WIDTH: f64 = 0.0;
const DEFAULT_COUPLING_STRENGTH: f64 = 1.0;
const DEFAULT_COUPLING_PHASE_BIAS: f64 = 0.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EulerPlane {
    axis: [f64; 3],
}

impl EulerPlane {
    pub fn new(axis: [f64; 3]) -> Result<Self, String> {
        let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        if norm < CELL_EPSILON {
            return Err("Euler plane axis must be non-zero".into());
        }
        let normalized = [axis[0] / norm, axis[1] / norm, axis[2] / norm];
        Ok(Self {
            axis: canonicalize_axis(normalized),
        })
    }

    pub fn i() -> Self {
        Self { axis: [1.0, 0.0, 0.0] }
    }

    pub fn j() -> Self {
        Self { axis: [0.0, 1.0, 0.0] }
    }

    pub fn k() -> Self {
        Self { axis: [0.0, 0.0, 1.0] }
    }

    pub fn from_quaternion(q: &[f64; 4]) -> Result<Self, String> {
        let imag = [q[1], q[2], q[3]];
        let norm = (imag[0] * imag[0] + imag[1] * imag[1] + imag[2] * imag[2]).sqrt();
        if norm < CELL_EPSILON {
            return Err(format!(
                "cannot infer a plane from boundary quaternion {:?}; plane must be carried explicitly",
                q
            ));
        }
        Self::new([imag[0] / norm, imag[1] / norm, imag[2] / norm])
    }

    pub fn axis(self) -> [f64; 3] {
        self.axis
    }

    pub fn matches(self, other: Self) -> bool {
        axis_distance(self.axis, other.axis) < CELL_EPSILON
    }

    pub fn relation(self, other: Self) -> PlaneRelation {
        let dot = self.axis[0] * other.axis[0] + self.axis[1] * other.axis[1] + self.axis[2] * other.axis[2];
        if self.matches(other) {
            PlaneRelation::Same
        } else if dot.abs() < CELL_EPSILON {
            PlaneRelation::Orthogonal
        } else {
            PlaneRelation::Oblique(dot)
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VerificationLandmark {
    Identity,
    Distinction,
    Intermediate,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PlaneRelation {
    Same,
    Orthogonal,
    Oblique(f64),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TwistSheet {
    Direct,
    Inverted,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SheetRelation {
    Same,
    Flipped,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CouplingState {
    strength: f64,
    phase_bias: f64,
}

impl CouplingState {
    pub fn new(strength: f64, phase_bias: f64) -> Result<Self, String> {
        if !strength.is_finite() || !(0.0..=1.0).contains(&strength) {
            return Err(format!("coupling strength must be finite and in [0,1], got {strength}"));
        }
        if !phase_bias.is_finite() {
            return Err(format!("coupling phase bias must be finite, got {phase_bias}"));
        }
        Ok(Self {
            strength,
            phase_bias: normalize_phase(phase_bias),
        })
    }

    pub fn neutral() -> Self {
        Self {
            strength: DEFAULT_COUPLING_STRENGTH,
            phase_bias: DEFAULT_COUPLING_PHASE_BIAS,
        }
    }

    pub fn strength(self) -> f64 {
        self.strength
    }

    pub fn phase_bias(self) -> f64 {
        self.phase_bias
    }

    pub fn bind(self, other: Self) -> Self {
        let strength = (self.strength * other.strength).sqrt();
        let bias = normalize_phase((self.phase_bias + other.phase_bias) * 0.5);
        Self { strength, phase_bias: bias }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NeighborCoupling {
    pub plane_relation: PlaneRelation,
    pub phase_offset: f64,
    pub sheet_relation: SheetRelation,
    pub coherence_overlap: f64,
    pub effective_strength: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VerificationCell {
    plane: EulerPlane,
    phase: f64,
    turns: i64,
    sheet: TwistSheet,
    coherence_width: f64,
    coupling: CouplingState,
}

impl VerificationCell {
    pub fn new(plane: EulerPlane, phase: f64) -> Self {
        Self::from_total_phase(plane, phase)
    }

    pub fn from_total_phase(plane: EulerPlane, total_phase: f64) -> Self {
        Self::from_total_phase_with_state(
            plane,
            total_phase,
            DEFAULT_COHERENCE_WIDTH,
            CouplingState::neutral(),
        )
    }

    pub fn from_total_phase_with_state(
        plane: EulerPlane,
        total_phase: f64,
        coherence_width: f64,
        coupling: CouplingState,
    ) -> Self {
        let turns = total_phase.div_euclid(TAU) as i64;
        let phase = total_phase.rem_euclid(TAU);
        let sheet = if turns.rem_euclid(2) == 0 {
            TwistSheet::Direct
        } else {
            TwistSheet::Inverted
        };
        Self {
            plane,
            phase,
            turns,
            sheet,
            coherence_width: sanitize_coherence_width(coherence_width),
            coupling,
        }
    }

    pub fn from_phase_and_turns(plane: EulerPlane, phase: f64, turns: i64) -> Self {
        let total_phase = phase + (turns as f64) * TAU;
        Self::from_total_phase(plane, total_phase)
    }

    pub fn from_phase_turns_and_state(
        plane: EulerPlane,
        phase: f64,
        turns: i64,
        coherence_width: f64,
        coupling: CouplingState,
    ) -> Self {
        let total_phase = phase + (turns as f64) * TAU;
        Self::from_total_phase_with_state(plane, total_phase, coherence_width, coupling)
    }

    pub fn identity(plane: EulerPlane) -> Self {
        Self::from_phase_and_turns(plane, 0.0, 0)
    }

    pub fn distinction(plane: EulerPlane) -> Self {
        Self::from_phase_and_turns(plane, PI, 0)
    }

    pub fn returned(plane: EulerPlane) -> Self {
        Self::from_phase_and_turns(plane, 0.0, 1)
    }

    pub fn verified_return(plane: EulerPlane) -> Self {
        Self::from_phase_and_turns(plane, 0.0, 2)
    }

    pub fn plane(self) -> EulerPlane {
        self.plane
    }

    pub fn phase(self) -> f64 {
        self.phase
    }

    pub fn total_phase(self) -> f64 {
        self.phase + (self.turns as f64) * TAU
    }

    pub fn turns(self) -> i64 {
        self.turns
    }

    pub fn sheet(self) -> TwistSheet {
        self.sheet
    }

    pub fn coherence_width(self) -> f64 {
        self.coherence_width
    }

    pub fn coherence(self) -> f64 {
        1.0 / (1.0 + self.coherence_width)
    }

    pub fn coupling(self) -> CouplingState {
        self.coupling
    }

    pub fn normalized_phase(self) -> f64 {
        normalize_phase(self.phase)
    }

    pub fn direction(self) -> f64 {
        self.total_phase().signum()
    }

    pub fn completed_turns(self) -> i64 {
        self.turns
    }

    pub fn geometry(self) -> [f64; 4] {
        let [x, y, z] = self.plane.axis();
        let s = self.phase.sin();
        [self.phase.cos(), x * s, y * s, z * s]
    }

    pub fn from_geometry_on_plane(plane: EulerPlane, q: &[f64; 4]) -> Result<Self, String> {
        let imag = [q[1], q[2], q[3]];
        let imag_norm = (imag[0] * imag[0] + imag[1] * imag[1] + imag[2] * imag[2]).sqrt();

        if imag_norm < CELL_EPSILON {
            if (q[0] - 1.0).abs() < CELL_EPSILON {
                return Ok(Self::from_phase_and_turns(plane, 0.0, 0));
            }
            if (q[0] + 1.0).abs() < CELL_EPSILON {
                return Ok(Self::from_phase_and_turns(plane, PI, 0));
            }
            return Err(format!("quaternion {:?} is not on S^3 close enough to a boundary rotor", q));
        }

        let axis = [imag[0] / imag_norm, imag[1] / imag_norm, imag[2] / imag_norm];
        let plane_axis = plane.axis();
        let aligned = axis[0] * plane_axis[0] + axis[1] * plane_axis[1] + axis[2] * plane_axis[2];
        if aligned.abs() < 1.0 - 1e-8 {
            return Err(format!(
                "quaternion {:?} does not lie on the requested Euler plane {:?}",
                q,
                plane.axis()
            ));
        }

        let signed_imag = imag[0] * plane_axis[0] + imag[1] * plane_axis[1] + imag[2] * plane_axis[2];
        let phase = signed_imag.atan2(q[0]);
        Ok(Self::from_phase_and_turns(plane, phase, 0))
    }

    pub fn from_geometry(q: &[f64; 4]) -> Result<Self, String> {
        let plane = EulerPlane::from_quaternion(q)?;
        Self::from_geometry_on_plane(plane, q)
    }

    pub fn advance(self, delta_phase: f64) -> Self {
        Self::from_total_phase_with_state(
            self.plane,
            self.total_phase() + delta_phase,
            self.coherence_width,
            self.coupling,
        )
    }

    pub fn compose(self, other: Self) -> Result<Self, String> {
        let bound_width = bind_coherence_width(self.coherence_width, other.coherence_width);
        let bound_coupling = self.coupling.bind(other.coupling);
        if self.plane.matches(other.plane) {
            return Ok(Self::from_total_phase_with_state(
                self.plane,
                self.total_phase() + other.total_phase(),
                bound_width,
                bound_coupling,
            ));
        }

        let q = compose(&self.geometry(), &other.geometry());
        if let Ok(decoded) = Self::from_geometry(&q) {
            return Ok(Self::from_phase_turns_and_state(
                decoded.plane(),
                decoded.phase(),
                self.turns + other.turns,
                bound_width,
                bound_coupling,
            ));
        }

        Err(format!(
            "composition landed on a boundary rotor {:?}; plane cannot be inferred without carrying it explicitly",
            q
        ))
    }

    pub fn landmark(self) -> VerificationLandmark {
        let phase = self.normalized_phase();
        if phase.abs() < CELL_EPSILON || (TAU - phase).abs() < CELL_EPSILON {
            VerificationLandmark::Identity
        } else if (phase - PI).abs() < CELL_EPSILON {
            VerificationLandmark::Distinction
        } else {
            VerificationLandmark::Intermediate
        }
    }

    pub fn distinction_crossings_to(self, next: Self) -> Result<i64, String> {
        ensure_same_plane(self, next)?;
        Ok(count_periodic_boundaries(
            self.total_phase(),
            next.total_phase(),
            PI,
            TAU,
        ))
    }

    pub fn return_crossings_to(self, next: Self) -> Result<i64, String> {
        ensure_same_plane(self, next)?;
        Ok(count_periodic_boundaries(
            self.total_phase(),
            next.total_phase(),
            0.0,
            TAU,
        ))
    }

    pub fn plane_relation(self, other: Self) -> PlaneRelation {
        self.plane.relation(other.plane)
    }

    pub fn sheet_relation(self, other: Self) -> SheetRelation {
        if self.sheet == other.sheet {
            SheetRelation::Same
        } else {
            SheetRelation::Flipped
        }
    }

    pub fn coupling_to(self, other: Self) -> NeighborCoupling {
        let phase_offset = normalize_phase(other.total_phase() - self.total_phase());
        let coherence_overlap = 1.0 / (1.0 + self.coherence_width + other.coherence_width);
        let effective_strength = coherence_overlap * self.coupling.bind(other.coupling).strength();
        NeighborCoupling {
            plane_relation: self.plane_relation(other),
            phase_offset,
            sheet_relation: self.sheet_relation(other),
            coherence_overlap,
            effective_strength,
        }
    }

    pub fn with_coherence_width(self, coherence_width: f64) -> Self {
        Self {
            coherence_width: sanitize_coherence_width(coherence_width),
            ..self
        }
    }

    pub fn bind_coherence(self, other: Self) -> Self {
        Self {
            coherence_width: bind_coherence_width(self.coherence_width, other.coherence_width),
            coupling: self.coupling.bind(other.coupling),
            ..self
        }
    }

    pub fn with_coupling(self, coupling: CouplingState) -> Self {
        Self { coupling, ..self }
    }

    pub fn is_plain_identity(self) -> bool {
        matches!(self.landmark(), VerificationLandmark::Identity) && self.turns == 0
    }
}

fn ensure_same_plane(left: VerificationCell, right: VerificationCell) -> Result<(), String> {
    if left.plane.matches(right.plane) {
        Ok(())
    } else {
        Err(format!(
            "cells live on different planes {:?} vs {:?}",
            left.plane.axis(),
            right.plane.axis()
        ))
    }
}

fn normalize_phase(phase: f64) -> f64 {
    phase.rem_euclid(TAU)
}

fn sanitize_coherence_width(width: f64) -> f64 {
    if !width.is_finite() || width < 0.0 {
        0.0
    } else {
        width
    }
}

fn bind_coherence_width(left: f64, right: f64) -> f64 {
    (left * left + right * right).sqrt()
}

fn count_periodic_boundaries(start: f64, end: f64, offset: f64, period: f64) -> i64 {
    if (end - start).abs() < CELL_EPSILON {
        return 0;
    }

    if end > start {
        (((end - offset) / period).floor() - ((start - offset) / period).floor()) as i64
    } else {
        -((((start - offset) / period).floor() - ((end - offset) / period).floor()) as i64)
    }
}

fn canonicalize_axis(axis: [f64; 3]) -> [f64; 3] {
    if axis[0] < -CELL_EPSILON
        || (axis[0].abs() < CELL_EPSILON && axis[1] < -CELL_EPSILON)
        || (axis[0].abs() < CELL_EPSILON && axis[1].abs() < CELL_EPSILON && axis[2] < -CELL_EPSILON)
    {
        [-axis[0], -axis[1], -axis[2]]
    } else {
        axis
    }
}

fn axis_distance(left: [f64; 3], right: [f64; 3]) -> f64 {
    ((left[0] - right[0]).powi(2) + (left[1] - right[1]).powi(2) + (left[2] - right[2]).powi(2)).sqrt()
}

pub fn identity_geometry() -> [f64; 4] {
    IDENTITY
}
