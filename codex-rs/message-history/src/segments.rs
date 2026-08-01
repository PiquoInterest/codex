//! Segment layout for the cross-session message history log.
//!
//! The logical history is stored as one active `history.jsonl` file plus zero
//! or more immutable rotated segments named `history.<start>.jsonl`, where
//! `<start>` is the global zero-based entry index of the segment's first line.
//! Appends only ever touch the active file. When the active file reaches the
//! rotation threshold it is renamed into place as the newest segment (which
//! preserves its identity for cached lookups) and a fresh active file begins.
//! Cap enforcement deletes whole segments oldest-first and never rewrites
//! retained bytes.
//!
//! Older codex versions that predate segmentation keep working against the
//! active file alone: they read and append the newest entries and simply do
//! not see rotated segments. Their legacy in-place trim of the active file is
//! tolerated; global counts self-correct on the next scan here because the
//! newest segment's entry count is always derived by reading it, never cached.

use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::path::Path;
use std::path::PathBuf;

const SEGMENT_PREFIX: &str = "history.";
const SEGMENT_SUFFIX: &str = ".jsonl";

/// One rotated, effectively immutable history segment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Segment {
    /// Global entry index of this segment's first line.
    pub(crate) start: u64,
    pub(crate) path: PathBuf,
}

/// Sorted view of the rotated segments beside an active history file.
#[derive(Debug, Default)]
pub(crate) struct SegmentSet {
    /// Ascending by `start`.
    pub(crate) segments: Vec<Segment>,
}

impl SegmentSet {
    /// Scans `dir` for `history.<digits>.jsonl` files.
    ///
    /// Unparseable names are ignored rather than treated as errors so foreign
    /// files cannot break history reads.
    pub(crate) fn scan(dir: &Path) -> Self {
        let mut segments = Vec::new();
        let Ok(entries) = std::fs::read_dir(dir) else {
            return Self::default();
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let Some(name) = name.to_str() else {
                continue;
            };
            let Some(start) = parse_segment_start(name) else {
                continue;
            };
            segments.push(Segment {
                start,
                path: entry.path(),
            });
        }
        segments.sort_by_key(|segment| segment.start);
        Self { segments }
    }

    /// Global entry index of the active file's first line.
    ///
    /// Derived as the newest segment's start plus its entry count; counting by
    /// reading (instead of trusting the next name) lets stray appends from
    /// pre-segmentation writers self-correct into the numbering.
    pub(crate) fn active_start(&self) -> u64 {
        let Some(newest) = self.segments.last() else {
            return 0;
        };
        newest.start.saturating_add(count_lines(&newest.path))
    }

    /// Entry count of `segment`, using name arithmetic for all but the newest
    /// segment and a bounded read for the newest.
    pub(crate) fn segment_count(&self, index: usize) -> u64 {
        match self.segments.get(index.saturating_add(1)) {
            Some(next) => next.start.saturating_sub(self.segments[index].start),
            None => self
                .segments
                .get(index)
                .map(|segment| count_lines(&segment.path))
                .unwrap_or(0),
        }
    }

    /// Locates the segment containing global entry `offset`, returning the
    /// segment index and the offset local to that segment's file.
    pub(crate) fn locate(&self, offset: u64) -> Option<(usize, u64)> {
        let index = self
            .segments
            .partition_point(|segment| segment.start <= offset)
            .checked_sub(1)?;
        let local = offset - self.segments[index].start;
        (local < self.segment_count(index)).then_some((index, local))
    }

    /// Total bytes across all rotated segments.
    pub(crate) fn total_bytes(&self) -> u64 {
        self.segments
            .iter()
            .filter_map(|segment| std::fs::metadata(&segment.path).ok())
            .map(|meta| meta.len())
            .fold(0u64, u64::saturating_add)
    }
}

/// Parses `history.<digits>.jsonl` into the segment start index.
fn parse_segment_start(name: &str) -> Option<u64> {
    let rest = name.strip_prefix(SEGMENT_PREFIX)?;
    let digits = rest.strip_suffix(SEGMENT_SUFFIX)?;
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

/// Returns the path a rotation of the active file must rename it to.
pub(crate) fn segment_path(dir: &Path, start: u64) -> PathBuf {
    dir.join(format!("{SEGMENT_PREFIX}{start}{SEGMENT_SUFFIX}"))
}

/// Counts newline-terminated entries plus a final unterminated one, matching
/// the `lines()` semantics used by lookups.
pub(crate) fn count_lines(path: &Path) -> u64 {
    let Ok(file) = File::open(path) else {
        return 0;
    };
    count_lines_of(file)
}

pub(crate) fn count_lines_of(file: File) -> u64 {
    let mut reader = BufReader::with_capacity(64 * 1024, file);
    let mut count: u64 = 0;
    let mut ends_with_newline = true;
    let mut saw_any = false;
    loop {
        let buffer = match reader.fill_buf() {
            Ok(buffer) => buffer,
            Err(_) => return count,
        };
        if buffer.is_empty() {
            break;
        }
        saw_any = true;
        count = count.saturating_add(bytecount(buffer, b'\n') as u64);
        ends_with_newline = buffer.last() == Some(&b'\n');
        let len = buffer.len();
        reader.consume(len);
    }
    if saw_any && !ends_with_newline {
        count = count.saturating_add(1);
    }
    count
}

fn bytecount(haystack: &[u8], needle: u8) -> usize {
    memchr::memchr_iter(needle, haystack).count()
}

/// Reads the zero-based `local_offset`th line of `file`.
pub(crate) fn read_line_at(file: File, local_offset: u64) -> Option<String> {
    let reader = BufReader::new(file);
    let mut lines = reader.lines();
    for _ in 0..local_offset {
        lines.next()?.ok()?;
    }
    lines.next()?.ok()
}

/// Deletes oldest segments until rotated bytes fit within `budget`.
///
/// Never touches the active file; the caller decides the budget (typically
/// `max_bytes` minus the active file's length, floored at zero).
pub(crate) fn delete_oldest_over_budget(set: &SegmentSet, budget: u64) {
    let mut total = set.total_bytes();
    // The newest segment is never deleted: it holds the most recently rotated
    // entries, and retaining the newest entry is an invariant carried over
    // from the previous in-place trim.
    let deletable = set.segments.len().saturating_sub(1);
    for segment in &set.segments[..deletable] {
        if total <= budget {
            break;
        }
        let len = std::fs::metadata(&segment.path)
            .map(|meta| meta.len())
            .unwrap_or(0);
        if std::fs::remove_file(&segment.path).is_ok() {
            total = total.saturating_sub(len);
        }
    }
}
