use std::f64::consts::PI;
use std::io;

use crate::arithmetic::{FullAddOutput, VerificationArithmetic};
use crate::cell::{EulerPlane, VerificationCell};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LogicObservation {
    pub plane: EulerPlane,
    pub total_phase: f64,
    pub remainder_phase: f64,
    pub completed_cycles: i64,
    pub state: VerificationCell,
    pub output: VerificationCell,
}

pub struct VerificationLogic;

impl VerificationLogic {
    pub fn xor(left: VerificationCell, right: VerificationCell) -> io::Result<LogicObservation> {
        let add = VerificationArithmetic::full_add(left, right, VerificationArithmetic::zero(shared_plane(left, right)?))?;
        Ok(LogicObservation {
            plane: add.state.plane(),
            total_phase: add.total_phase,
            remainder_phase: add.state.phase(),
            completed_cycles: add.completed_cycles,
            state: add.state,
            output: add.sum,
        })
    }

    pub fn and(left: VerificationCell, right: VerificationCell) -> io::Result<LogicObservation> {
        let add = VerificationArithmetic::full_add(left, right, VerificationArithmetic::zero(shared_plane(left, right)?))?;
        Ok(LogicObservation {
            plane: add.state.plane(),
            total_phase: add.total_phase,
            remainder_phase: add.state.phase(),
            completed_cycles: add.completed_cycles,
            state: add.state,
            output: add.carry,
        })
    }

    pub fn not(input: VerificationCell) -> LogicObservation {
        let output = input.advance(PI);
        let total_phase = input.total_phase() + PI;
        LogicObservation {
            plane: input.plane(),
            total_phase,
            remainder_phase: output.phase(),
            completed_cycles: output.completed_turns(),
            state: output,
            output,
        }
    }

    pub fn or(left: VerificationCell, right: VerificationCell) -> io::Result<LogicObservation> {
        let xor = Self::xor(left, right)?;
        let and = Self::and(left, right)?;
        let add = VerificationArithmetic::full_add(xor.output, and.output, VerificationArithmetic::zero(xor.plane))?;
        Ok(LogicObservation {
            plane: add.state.plane(),
            total_phase: add.total_phase,
            remainder_phase: add.state.phase(),
            completed_cycles: add.completed_cycles,
            state: add.state,
            output: add.sum,
        })
    }

    pub fn from_full_add(add: FullAddOutput) -> LogicObservation {
        LogicObservation {
            plane: add.state.plane(),
            total_phase: add.total_phase,
            remainder_phase: add.state.phase(),
            completed_cycles: add.completed_cycles,
            state: add.state,
            output: add.sum,
        }
    }
}

fn shared_plane(left: VerificationCell, right: VerificationCell) -> io::Result<EulerPlane> {
    if left.plane().matches(right.plane()) {
        Ok(left.plane())
    } else {
        Err(io::Error::other(format!(
            "logic operation requires a shared Euler plane, got {:?} and {:?}",
            left.plane().axis(),
            right.plane().axis()
        )))
    }
}
