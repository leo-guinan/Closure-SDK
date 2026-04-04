use closure_rs::groups::sphere::{sphere_compose as compose, IDENTITY};

use crate::cell::{EulerPlane, VerificationCell};

#[derive(Clone, Debug, PartialEq)]
pub struct VerificationWord {
    cells_le: Vec<VerificationCell>,
}

impl VerificationWord {
    pub fn new(cells_le: Vec<VerificationCell>) -> Self {
        Self { cells_le }
    }

    pub fn single(cell: VerificationCell) -> Self {
        Self { cells_le: vec![cell] }
    }

    pub fn identity_run(plane: EulerPlane, len: usize) -> Self {
        Self {
            cells_le: (0..len).map(|_| VerificationCell::identity(plane)).collect(),
        }
    }

    pub fn from_phases_on_plane(plane: EulerPlane, phases_le: &[f64]) -> Self {
        Self {
            cells_le: phases_le
                .iter()
                .map(|phase| VerificationCell::new(plane, *phase))
                .collect(),
        }
    }

    pub fn len(&self) -> usize {
        self.cells_le.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cells_le.is_empty()
    }

    pub fn cell(&self, idx: usize) -> Option<VerificationCell> {
        self.cells_le.get(idx).copied()
    }

    pub fn cells_le(&self) -> &[VerificationCell] {
        &self.cells_le
    }

    pub fn geometry(&self) -> [f64; 4] {
        self.cells_le
            .iter()
            .fold(IDENTITY, |state, cell| compose(&state, &cell.geometry()))
    }

    pub fn phase_signature(&self) -> Vec<(usize, f64)> {
        self.cells_le
            .iter()
            .enumerate()
            .map(|(idx, cell)| (idx, cell.phase()))
            .collect()
    }

    pub fn turn_signature(&self) -> Vec<(usize, i64)> {
        self.cells_le
            .iter()
            .enumerate()
            .map(|(idx, cell)| (idx, cell.turns()))
            .collect()
    }

    pub fn coherence_signature(&self) -> Vec<(usize, f64)> {
        self.cells_le
            .iter()
            .enumerate()
            .map(|(idx, cell)| (idx, cell.coherence_width()))
            .collect()
    }

    pub fn coupling_signature(&self) -> Vec<(usize, f64, f64)> {
        self.cells_le
            .iter()
            .enumerate()
            .map(|(idx, cell)| (idx, cell.coupling().strength(), cell.coupling().phase_bias()))
            .collect()
    }
}
