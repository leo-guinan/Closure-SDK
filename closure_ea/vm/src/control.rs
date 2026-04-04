use std::io;

use crate::cell::NeighborCoupling;
use crate::word::VerificationWord;

#[derive(Clone, Debug, PartialEq)]
pub struct ComparisonObservation {
    pub local_couplings: Vec<NeighborCoupling>,
    pub comparable: bool,
    pub note: &'static str,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BranchObservation {
    pub admissible: bool,
    pub note: &'static str,
}

pub struct VerificationControl;

impl VerificationControl {
    pub fn compare_words(_left: &VerificationWord, _right: &VerificationWord) -> io::Result<ComparisonObservation> {
        Err(io::Error::other(
            "compare is intentionally blocked until coherence width and neighbor coupling are promoted into the control law",
        ))
    }

    pub fn branch_on(_comparison: &ComparisonObservation) -> io::Result<BranchObservation> {
        Err(io::Error::other(
            "branch is intentionally blocked until compare is re-derived from the richer rotor carrier",
        ))
    }
}
