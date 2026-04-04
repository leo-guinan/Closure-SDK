use std::io;

use closure_rs::table::{ColumnDef, ColumnType, ColumnValue, Table};

use crate::cell::{CouplingState, EulerPlane, VerificationCell};
use crate::word::VerificationWord;

pub struct WordMemory;
const FREE_WORD_ID: &[u8] = b"__free_verification_cell__";

impl WordMemory {
    pub fn table_schema() -> Vec<ColumnDef> {
        let bytes_col = |name: &str, indexed: bool| ColumnDef {
            name: name.into(),
            col_type: ColumnType::Bytes,
            indexed,
            not_null: true,
            unique: false,
        };
        let f64_col = |name: &str| ColumnDef {
            name: name.into(),
            col_type: ColumnType::F64,
            indexed: false,
            not_null: true,
            unique: false,
        };

        vec![
            bytes_col("word_id", true),
            ColumnDef {
                name: "cell_index".into(),
                col_type: ColumnType::I64,
                indexed: false,
                not_null: true,
                unique: false,
            },
            f64_col("plane_x"),
            f64_col("plane_y"),
            f64_col("plane_z"),
            f64_col("phase"),
            ColumnDef {
                name: "turns".into(),
                col_type: ColumnType::I64,
                indexed: false,
                not_null: true,
                unique: false,
            },
            f64_col("coherence_width"),
            f64_col("coupling_strength"),
            f64_col("coupling_phase_bias"),
        ]
    }

    pub fn save_word(table: &mut Table, word_id: &[u8], word: &VerificationWord) -> io::Result<usize> {
        let mut existing_rows = table.filter_equals("word_id", word_id)?;
        existing_rows.sort_by_key(|&row| match table.get_row(row) {
            Ok(data) => match data.get(1) {
                Some(ColumnValue::I64(idx)) => *idx,
                _ => i64::MAX,
            },
            Err(_) => i64::MAX,
        });

        let cells = word.cells_le();
        let mut free_rows = table.filter_equals("word_id", FREE_WORD_ID)?;
        free_rows.sort_unstable();

        for (cell_index, cell) in cells.iter().enumerate() {
            let [x, y, z] = cell.plane().axis();
            let row_data = [
                ColumnValue::Bytes(word_id.to_vec()),
                ColumnValue::I64(cell_index as i64),
                ColumnValue::F64(x),
                ColumnValue::F64(y),
                ColumnValue::F64(z),
                ColumnValue::F64(cell.phase()),
                ColumnValue::I64(cell.turns()),
                ColumnValue::F64(cell.coherence_width()),
                ColumnValue::F64(cell.coupling().strength()),
                ColumnValue::F64(cell.coupling().phase_bias()),
            ];

            if let Some(&row) = existing_rows.get(cell_index) {
                table.update(row, &row_data)?;
            } else if let Some(row) = free_rows.pop() {
                table.update(row, &row_data)?;
            } else {
                table.insert(&row_data)?;
            }
        }

        let free_plane = EulerPlane::i().axis();
        for &row in existing_rows.iter().skip(cells.len()) {
            table.update(
                row,
                &[
                    ColumnValue::Bytes(FREE_WORD_ID.to_vec()),
                    ColumnValue::I64(0),
                    ColumnValue::F64(free_plane[0]),
                    ColumnValue::F64(free_plane[1]),
                    ColumnValue::F64(free_plane[2]),
                    ColumnValue::F64(0.0),
                    ColumnValue::I64(0),
                    ColumnValue::F64(0.0),
                    ColumnValue::F64(1.0),
                    ColumnValue::F64(0.0),
                ],
            )?;
        }

        Ok(cells.len())
    }

    pub fn load_word(table: &mut Table, word_id: &[u8]) -> io::Result<VerificationWord> {
        let mut rows = table.filter_equals("word_id", word_id)?;
        if rows.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("word {:?} not found", String::from_utf8_lossy(word_id)),
            ));
        }

        let mut indexed_cells = Vec::with_capacity(rows.len());
        for row in rows.drain(..) {
            let data = table.get_row(row)?;
            let cell_index = match data.get(1) {
                Some(ColumnValue::I64(idx)) if *idx >= 0 => *idx as usize,
                other => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("cell_index must be non-negative i64, got {other:?}"),
                    ))
                }
            };
            let plane = EulerPlane::new([
                read_f64(&data, 2)?,
                read_f64(&data, 3)?,
                read_f64(&data, 4)?,
            ])
            .map_err(io::Error::other)?;
            let phase = read_f64(&data, 5)?;
            let turns = read_i64(&data, 6)?;
            let coherence_width = read_f64(&data, 7)?;
            let coupling = CouplingState::new(read_f64(&data, 8)?, read_f64(&data, 9)?)
                .map_err(io::Error::other)?;
            indexed_cells.push((
                cell_index,
                VerificationCell::from_phase_turns_and_state(plane, phase, turns, coherence_width, coupling),
            ));
        }

        indexed_cells.sort_by_key(|(idx, _)| *idx);
        let mut cells_le = Vec::with_capacity(indexed_cells.len());
        for (expected_index, (cell_index, cell)) in indexed_cells.into_iter().enumerate() {
            if cell_index != expected_index {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "word cells are not contiguous from zero: expected {expected_index}, got {cell_index}"
                    ),
                ));
            }
            cells_le.push(cell);
        }

        Ok(VerificationWord::new(cells_le))
    }
}

fn read_f64(row: &[ColumnValue], idx: usize) -> io::Result<f64> {
    match row.get(idx) {
        Some(ColumnValue::F64(value)) => Ok(*value),
        other => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("expected f64 at column {idx}, got {other:?}"),
        )),
    }
}

fn read_i64(row: &[ColumnValue], idx: usize) -> io::Result<i64> {
    match row.get(idx) {
        Some(ColumnValue::I64(value)) => Ok(*value),
        other => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("expected i64 at column {idx}, got {other:?}"),
        )),
    }
}
