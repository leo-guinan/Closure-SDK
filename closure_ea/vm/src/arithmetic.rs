use std::f64::consts::{PI, TAU};
use std::io;

use crate::cell::{CouplingState, EulerPlane, VerificationCell, VerificationLandmark};
use crate::word::VerificationWord;

const PHASE_EPSILON: f64 = 1e-9;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FullAddOutput {
    pub total_phase: f64,
    pub state: VerificationCell,
    pub sum: VerificationCell,
    pub carry: VerificationCell,
    pub completed_cycles: i64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FullSubtractOutput {
    pub raw_phase: f64,
    pub state: VerificationCell,
    pub difference: VerificationCell,
    pub borrow: VerificationCell,
    pub borrow_cycles: i64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WordSubtractionResult {
    pub difference: VerificationWord,
    pub borrow_out: VerificationCell,
}

pub struct VerificationArithmetic;

impl VerificationArithmetic {
    pub fn zero(plane: EulerPlane) -> VerificationCell {
        VerificationCell::identity(plane)
    }

    pub fn one(plane: EulerPlane) -> VerificationCell {
        VerificationCell::distinction(plane)
    }

    pub fn full_add(
        left: VerificationCell,
        right: VerificationCell,
        carry_in: VerificationCell,
    ) -> io::Result<FullAddOutput> {
        let plane = shared_plane(&[left, right, carry_in]).map_err(io::Error::other)?;
        let total_phase = contribution_phase(left)? + contribution_phase(right)? + contribution_phase(carry_in)?;
        let completed_cycles = (total_phase / TAU).floor() as i64;
        let remainder = normalize_binary_remainder(total_phase - (completed_cycles as f64) * TAU)?;
        let (coherence_width, coupling) = aggregate_state(&[left, right, carry_in]);
        let state = VerificationCell::from_total_phase_with_state(plane, total_phase, coherence_width, coupling);

        Ok(FullAddOutput {
            total_phase,
            state,
            sum: cell_from_binary_remainder(state, remainder),
            carry: carry_signal(state, completed_cycles)?,
            completed_cycles,
        })
    }

    pub fn full_subtract(
        left: VerificationCell,
        right: VerificationCell,
        borrow_in: VerificationCell,
    ) -> io::Result<FullSubtractOutput> {
        let plane = shared_plane(&[left, right, borrow_in]).map_err(io::Error::other)?;
        let raw_phase = contribution_phase(left)? - contribution_phase(right)? - contribution_phase(borrow_in)?;
        let borrow_cycles = if raw_phase < -PHASE_EPSILON { 1 } else { 0 };
        let lifted = raw_phase + (borrow_cycles as f64) * TAU;
        let remainder = normalize_binary_remainder(lifted)?;
        let (coherence_width, coupling) = aggregate_state(&[left, right, borrow_in]);
        let state = VerificationCell::from_phase_turns_and_state(plane, remainder, borrow_cycles, coherence_width, coupling);

        Ok(FullSubtractOutput {
            raw_phase,
            state,
            difference: cell_from_binary_remainder(state, remainder),
            borrow: carry_signal(state, borrow_cycles)?,
            borrow_cycles,
        })
    }

    pub fn add_words(left: &VerificationWord, right: &VerificationWord) -> io::Result<VerificationWord> {
        let plane = shared_word_plane(left, right)?;
        let width = left.len().max(right.len());
        let mut out = Vec::with_capacity(width + 1);
        let mut carry = Self::zero(plane);

        for idx in 0..width {
            let left_cell = left.cell(idx).unwrap_or_else(|| Self::zero(plane));
            let right_cell = right.cell(idx).unwrap_or_else(|| Self::zero(plane));
            let step = Self::full_add(left_cell, right_cell, carry)?;
            out.push(step.state);
            carry = step.carry;
        }

        if contribution_phase(carry)? > PHASE_EPSILON {
            out.push(carry);
        }

        Ok(VerificationWord::new(trim_identity_tail(out)))
    }

    pub fn subtract_words(left: &VerificationWord, right: &VerificationWord) -> io::Result<WordSubtractionResult> {
        let plane = shared_word_plane(left, right)?;
        let width = left.len().max(right.len());
        let mut out = Vec::with_capacity(width);
        let mut borrow = Self::zero(plane);

        for idx in 0..width {
            let left_cell = left.cell(idx).unwrap_or_else(|| Self::zero(plane));
            let right_cell = right.cell(idx).unwrap_or_else(|| Self::zero(plane));
            let step = Self::full_subtract(left_cell, right_cell, borrow)?;
            out.push(step.state);
            borrow = step.borrow;
        }

        Ok(WordSubtractionResult {
            difference: VerificationWord::new(trim_identity_tail(out)),
            borrow_out: borrow,
        })
    }

    pub fn increment_word(word: &VerificationWord) -> io::Result<VerificationWord> {
        let plane = first_plane(word).ok_or_else(|| io::Error::other("cannot increment an empty verification word"))?;
        Self::add_words(word, &VerificationWord::single(Self::one(plane)))
    }

    pub fn decrement_word(word: &VerificationWord) -> io::Result<WordSubtractionResult> {
        let plane = first_plane(word).ok_or_else(|| io::Error::other("cannot decrement an empty verification word"))?;
        Self::subtract_words(word, &VerificationWord::single(Self::one(plane)))
    }
}

fn trim_identity_tail(mut cells: Vec<VerificationCell>) -> Vec<VerificationCell> {
    while cells.len() > 1 {
        let Some(last) = cells.last().copied() else { break; };
        if last.is_plain_identity() {
            cells.pop();
        } else {
            break;
        }
    }
    cells
}

fn first_plane(word: &VerificationWord) -> Option<EulerPlane> {
    word.cells_le().first().map(|cell| cell.plane())
}

fn shared_word_plane(left: &VerificationWord, right: &VerificationWord) -> io::Result<EulerPlane> {
    let plane = first_plane(left)
        .or_else(|| first_plane(right))
        .ok_or_else(|| io::Error::other("cannot infer arithmetic plane from two empty verification words"))?;

    ensure_word_plane(left, plane)?;
    ensure_word_plane(right, plane)?;
    Ok(plane)
}

fn ensure_word_plane(word: &VerificationWord, plane: EulerPlane) -> io::Result<()> {
    for cell in word.cells_le() {
        if !cell.plane().matches(plane) {
            return Err(io::Error::other(format!(
                "word mixes Euler planes {:?} and {:?}; binary arithmetic expects a shared plane",
                plane.axis(),
                cell.plane().axis()
            )));
        }
    }
    Ok(())
}

fn shared_plane(cells: &[VerificationCell]) -> Result<EulerPlane, String> {
    let plane = cells
        .first()
        .map(|cell| cell.plane())
        .ok_or_else(|| String::from("need at least one cell to infer a plane"))?;
    for cell in cells.iter().skip(1) {
        if !cell.plane().matches(plane) {
            return Err(format!(
                "cells live on different Euler planes {:?} vs {:?}",
                plane.axis(),
                cell.plane().axis()
            ));
        }
    }
    Ok(plane)
}

fn contribution_phase(cell: VerificationCell) -> io::Result<f64> {
    match cell.landmark() {
        VerificationLandmark::Identity => Ok(0.0),
        VerificationLandmark::Distinction => Ok(PI),
        VerificationLandmark::Intermediate => Err(io::Error::other(format!(
            "binary phase arithmetic expects identity or distinction landmarks, got total phase {} (local phase {}, turns {}) on plane {:?}",
            cell.total_phase(),
            cell.phase(),
            cell.turns(),
            cell.plane().axis()
        ))),
    }
}

fn normalize_binary_remainder(phase: f64) -> io::Result<f64> {
    if phase.abs() < PHASE_EPSILON || (phase - TAU).abs() < PHASE_EPSILON {
        Ok(0.0)
    } else if (phase - PI).abs() < PHASE_EPSILON {
        Ok(PI)
    } else {
        Err(io::Error::other(format!(
            "binary phase arithmetic expected remainder 0 or π, got {phase}"
        )))
    }
}

fn cell_from_binary_remainder(template: VerificationCell, remainder: f64) -> VerificationCell {
    if remainder.abs() < PHASE_EPSILON {
        VerificationCell::from_phase_turns_and_state(
            template.plane(),
            0.0,
            0,
            template.coherence_width(),
            template.coupling(),
        )
    } else {
        VerificationCell::from_phase_turns_and_state(
            template.plane(),
            PI,
            0,
            template.coherence_width(),
            template.coupling(),
        )
    }
}

fn carry_signal(template: VerificationCell, cycles: i64) -> io::Result<VerificationCell> {
    match cycles {
        0 => Ok(VerificationCell::from_phase_turns_and_state(
            template.plane(),
            0.0,
            0,
            template.coherence_width(),
            template.coupling(),
        )),
        1 => Ok(VerificationCell::from_phase_turns_and_state(
            template.plane(),
            PI,
            0,
            template.coherence_width(),
            template.coupling(),
        )),
        other => Err(io::Error::other(format!(
            "binary phase arithmetic expected 0 or 1 completed cycles, got {other}"
        ))),
    }
}

fn aggregate_state(cells: &[VerificationCell]) -> (f64, CouplingState) {
    let mut width_sq_sum = 0.0;
    let mut coupling = CouplingState::neutral();
    for (idx, cell) in cells.iter().enumerate() {
        width_sq_sum += cell.coherence_width().powi(2);
        coupling = if idx == 0 {
            cell.coupling()
        } else {
            coupling.bind(cell.coupling())
        };
    }
    (width_sq_sum.sqrt(), coupling)
}
