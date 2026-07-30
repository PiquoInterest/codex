//! Persistence layer for the global, append-only *message history* file.
//!
//! The history is stored at `~/.codex/history.jsonl` with **one JSON object per
//! line** so that it can be efficiently appended to and parsed with standard
//! JSON-Lines tooling. Each record has the following schema:
//!
//! ````text
//! {"session_id":"<uuid>","ts":<unix_seconds>,"text":"<message>"}
//! ````
//!
//! To minimize the chance of interleaved writes when multiple processes are
//! appending concurrently, callers should *prepare the full line* (record +
//! trailing `\n`) and write it with a **single `write(2)` system call** while
//! the file descriptor is opened with the `O_APPEND` flag. POSIX guarantees
//! that writes up to `PIPE_BUF` bytes are atomic in that case.

use std::fs::File;
use std::fs::OpenOptions;
use std::io::Read;
use std::io::Result;
use std::io::Seek;
use std::io::SeekFrom;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;

use memchr::memchr_iter;
use serde::Deserialize;
use serde::Serialize;

use std::time::Duration;
use tokio::fs;

use codex_config::types::History;
use codex_config::types::HistoryPersistence;

mod batch;
pub use batch::HistoryBatch;
pub use batch::HistoryBatchCursor;
pub use batch::HistoryBatchEntry;
pub use batch::lookup_batch;

#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

/// Filename that stores the message history inside `~/.codex`.
mod segments;
use segments::SegmentSet;

const HISTORY_FILENAME: &str = "history.jsonl";
const HISTORY_READ_BUFFER_SIZE: usize = 8192;
/// Larger buffer for the whole-file newline count at thread open; the batch
/// scanner keeps the smaller buffer, which its chunk-stitching tests and
/// stack-allocated read path are sized around.
const HISTORY_COUNT_BUFFER_SIZE: usize = 256 * 1024;

/// When history exceeds the hard cap, trim it down to this fraction of `max_bytes`.

const MAX_RETRIES: usize = 10;
const RETRY_SLEEP: Duration = Duration::from_millis(100);

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct HistoryEntry {
    pub session_id: String,
    pub ts: u64,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HistoryConfig {
    pub codex_home: PathBuf,
    pub persistence: HistoryPersistence,
    pub max_bytes: Option<usize>,
}

impl HistoryConfig {
    pub fn new(codex_home: impl Into<PathBuf>, history: &History) -> Self {
        Self {
            codex_home: codex_home.into(),
            persistence: history.persistence,
            max_bytes: history.max_bytes,
        }
    }
}

fn history_filepath(config: &HistoryConfig) -> PathBuf {
    config.codex_home.join(HISTORY_FILENAME)
}

/// Append a `text` entry associated with `conversation_id` to the history file.
///
/// Uses advisory file locking (`File::try_lock`) with a retry loop to ensure
/// concurrent writes from multiple TUI processes do not interleave. The lock
/// acquisition and write are performed inside `spawn_blocking` so the caller's
/// async runtime is not blocked.
///
/// The entry is silently skipped when `config.history.persistence` is
/// [`HistoryPersistence::None`].
///
/// # Errors
///
/// Returns an I/O error if the history file cannot be opened/created, the
/// system clock is before the Unix epoch, or the exclusive lock cannot be
/// acquired after [`MAX_RETRIES`] attempts.
pub async fn append_entry(
    text: &str,
    conversation_id: impl std::fmt::Display,
    config: &HistoryConfig,
) -> Result<()> {
    match config.persistence {
        HistoryPersistence::SaveAll => {
            // Save everything: proceed.
        }
        HistoryPersistence::None => {
            // No history persistence requested.
            return Ok(());
        }
    }

    // TODO: check `text` for sensitive patterns

    // Resolve `~/.codex/history.jsonl` and ensure the parent directory exists.
    let path = history_filepath(config);
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    // Compute timestamp (seconds since the Unix epoch).
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| std::io::Error::other(format!("system clock before Unix epoch: {e}")))?
        .as_secs();

    // Construct the JSON line first so we can write it in a single syscall.
    let entry = HistoryEntry {
        session_id: conversation_id.to_string(),
        ts,
        text: text.to_string(),
    };
    let mut line = serde_json::to_string(&entry)
        .map_err(|e| std::io::Error::other(format!("failed to serialise history entry: {e}")))?;
    line.push('\n');

    // Open the history file for read/write access (append-only on Unix).
    let mut options = OpenOptions::new();
    options.read(true).write(true).create(true);
    #[cfg(unix)]
    {
        options.append(true);
        options.mode(0o600);
    }

    let history_file = options.open(&path)?;

    // Ensure permissions.
    ensure_owner_only_permissions(&history_file).await?;
    drop(history_file);

    let history_max_bytes = config.max_bytes;

    // Perform a blocking write under an advisory write lock using std::fs.
    tokio::task::spawn_blocking(move || -> Result<()> {
        // Retry a few times to avoid indefinite blocking when contended. The
        // file is reopened on every attempt: a concurrent writer may rotate
        // the path between our open and lock, in which case the locked handle
        // would reference a frozen segment rather than the active file.
        for _ in 0..MAX_RETRIES {
            let mut options = OpenOptions::new();
            options.read(true).write(true).create(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.append(true);
                options.mode(0o600);
            }
            let mut history_file = options.open(&path)?;
            match history_file.try_lock() {
                Ok(()) => {
                    if !locked_handle_matches_path(&history_file, &path) {
                        // Lost a rotation race; retry against the fresh active file.
                        continue;
                    }
                    // While holding the exclusive lock, write the full line.
                    // We do not open the file with `append(true)` on Windows, so ensure the
                    // cursor is positioned at the end before writing.
                    history_file.seek(SeekFrom::End(0))?;
                    history_file.write_all(line.as_bytes())?;
                    history_file.flush()?;
                    maybe_rotate_and_enforce(&history_file, &path, history_max_bytes)?;
                    return Ok(());
                }
                Err(std::fs::TryLockError::WouldBlock) => {
                    std::thread::sleep(RETRY_SLEEP);
                }
                Err(e) => return Err(e.into()),
            }
        }

        Err(std::io::Error::new(
            std::io::ErrorKind::WouldBlock,
            "could not acquire exclusive lock on history file after multiple attempts",
        ))
    })
    .await??;

    Ok(())
}

/// Returns whether a locked handle still refers to the file at `path`.
///
/// A concurrent rotation renames the active file after another writer opened
/// it but before that writer acquired the lock; comparing identities detects
/// the swap. Platforms with no file identity treat every handle as current.
fn locked_handle_matches_path(file: &File, path: &Path) -> bool {
    let (Ok(handle_meta), Ok(path_meta)) = (file.metadata(), std::fs::metadata(path)) else {
        return false;
    };
    match (log_identity(&handle_meta), log_identity(&path_meta)) {
        (Some(handle_id), Some(path_id)) => handle_id == path_id,
        _ => true,
    }
}

/// Rotates a full active file into an immutable segment and enforces the cap.
///
/// Runs under the exclusive append lock. When the active file reaches a
/// quarter of `max_bytes` it is renamed to `history.<start>.jsonl` (keeping
/// its identity for cached lookups); the next append recreates the active
/// file. Cap enforcement then deletes whole segments oldest-first, never the
/// newest segment, so the most recent entry always survives.
fn maybe_rotate_and_enforce(file: &File, path: &Path, max_bytes: Option<usize>) -> Result<()> {
    let Some(max_bytes) = max_bytes else {
        return Ok(());
    };
    let rotate_threshold = (max_bytes as u64 / 4).max(1);
    if file.metadata()?.len() < rotate_threshold {
        return Ok(());
    }
    let Some(dir) = path.parent() else {
        return Ok(());
    };
    let set = SegmentSet::scan(dir);
    let start = set.active_start();
    std::fs::rename(path, segments::segment_path(dir, start))?;
    let set = SegmentSet::scan(dir);
    segments::delete_oldest_over_budget(&set, max_bytes as u64);
    Ok(())
}

/// Asynchronously fetch the history file's *identifier* and current entry count.
///
/// The identifier is the file's inode on Unix or creation time on Windows.
/// The entry count is derived by counting newline bytes in the file. Returns
/// `(0, 0)` when the file does not exist or its metadata cannot be read. If
/// metadata succeeds but the file cannot be opened or scanned, returns
/// `(log_id, 0)` so callers can still detect that a history file exists.
pub async fn history_metadata(config: &HistoryConfig) -> (u64, usize) {
    let path = history_filepath(config);
    let (log_id, active_count) = history_metadata_for_file(&path).await;
    let dir = path.parent().map(Path::to_path_buf);
    let start = tokio::task::spawn_blocking(move || {
        dir.map(|dir| SegmentSet::scan(&dir).active_start())
            .unwrap_or(0)
    })
    .await
    .unwrap_or(0);
    let total = usize::try_from(start)
        .unwrap_or(usize::MAX)
        .saturating_add(active_count);
    (log_id, total)
}

/// Look up a single history entry by file identity and zero-based offset.
///
/// Returns `Some(entry)` when the current history file's identifier (inode on
/// Unix, creation time on Windows) matches `log_id` **and** a valid JSON
/// record exists at `offset`. Returns `None` on any mismatch, I/O error, or
/// parse failure, all of which are logged at `warn` level.
///
/// This function is synchronous because it acquires a shared advisory file lock
/// via `File::try_lock_shared`. Callers on an async runtime should wrap it in
/// `spawn_blocking`.
pub fn lookup(log_id: u64, offset: usize, config: &HistoryConfig) -> Option<HistoryEntry> {
    let path = history_filepath(config);
    lookup_history_entry(&path, log_id, offset)
}

/// On Unix systems, ensure the file permissions are `0o600` (rw-------). If the
/// permissions cannot be changed the error is propagated to the caller.
#[cfg(unix)]
async fn ensure_owner_only_permissions(file: &File) -> Result<()> {
    let metadata = file.metadata()?;
    let current_mode = metadata.permissions().mode() & 0o777;
    if current_mode != 0o600 {
        let mut perms = metadata.permissions();
        perms.set_mode(0o600);
        let perms_clone = perms.clone();
        let file_clone = file.try_clone()?;
        tokio::task::spawn_blocking(move || file_clone.set_permissions(perms_clone)).await??;
    }
    Ok(())
}

#[cfg(windows)]
// On Windows, simply succeed.
async fn ensure_owner_only_permissions(_file: &File) -> Result<()> {
    Ok(())
}

async fn history_metadata_for_file(path: &Path) -> (u64, usize) {
    let log_id = match fs::metadata(path).await {
        Ok(metadata) => log_identity(&metadata).unwrap_or(0),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return (0, 0),
        Err(_) => return (0, 0),
    };

    let path = path.to_path_buf();
    let count = tokio::task::spawn_blocking(move || {
        let mut file = match File::open(path) {
            Ok(f) => f,
            Err(_) => return 0,
        };

        let mut buf = vec![0u8; HISTORY_COUNT_BUFFER_SIZE];
        let mut count = 0usize;
        loop {
            match file.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    count += memchr_iter(b'\n', &buf[..n]).count();
                }
                Err(_) => return 0,
            }
        }
        count
    })
    .await
    .unwrap_or(0);

    (log_id, count)
}

fn lookup_history_entry(path: &Path, log_id: u64, offset: usize) -> Option<HistoryEntry> {
    let dir = path.parent()?;
    let set = SegmentSet::scan(dir);

    let active = OpenOptions::new().read(true).open(path).ok();
    let active_id = active
        .as_ref()
        .and_then(|file| file.metadata().ok())
        .and_then(|metadata| log_identity(&metadata));

    if log_id == 0 || active_id == Some(log_id) || active_id.is_none() {
        // Global resolution: `offset` counts from the oldest retained entry
        // across rotated segments plus the active file.
        let start = set.active_start();
        let global = offset as u64;
        if global >= start {
            let local = usize::try_from(global - start).ok()?;
            return read_active_entry_locked(active?, local);
        }
        let (index, local) = set.locate(global)?;
        let file = File::open(&set.segments[index].path).ok()?;
        return parse_history_line(segments::read_line_at(file, local)?);
    }

    // The caller's identifier predates a rotation: it names what is now an
    // immutable segment, and `offset` is local to that file. Rotation renames
    // the file in place, so identity and line numbering both still hold.
    let segment = set.segments.iter().find(|segment| {
        std::fs::metadata(&segment.path)
            .ok()
            .and_then(|metadata| log_identity(&metadata))
            == Some(log_id)
    })?;
    let file = File::open(&segment.path).ok()?;
    parse_history_line(segments::read_line_at(file, offset as u64)?)
}

/// Reads the `local`th line of the active history file under a shared lock.
///
/// The lock is retried a bounded number of times; rotated segments are read
/// without locking because they are frozen after rotation.
fn read_active_entry_locked(file: File, local: usize) -> Option<HistoryEntry> {
    for _ in 0..MAX_RETRIES {
        match file.try_lock_shared() {
            Ok(()) => {
                return parse_history_line(segments::read_line_at(file, local as u64)?);
            }
            Err(std::fs::TryLockError::WouldBlock) => {
                std::thread::sleep(RETRY_SLEEP);
            }
            Err(e) => {
                tracing::warn!(error = %e, "failed to acquire shared lock on history file");
                return None;
            }
        }
    }
    None
}

fn parse_history_line(line: String) -> Option<HistoryEntry> {
    match serde_json::from_str::<HistoryEntry>(&line) {
        Ok(entry) => Some(entry),
        Err(e) => {
            tracing::warn!(error = %e, "failed to parse history entry");
            None
        }
    }
}

#[cfg(unix)]
fn log_identity(metadata: &std::fs::Metadata) -> Option<u64> {
    use std::os::unix::fs::MetadataExt;
    Some(metadata.ino())
}

#[cfg(windows)]
fn log_identity(metadata: &std::fs::Metadata) -> Option<u64> {
    use std::os::windows::fs::MetadataExt;
    Some(metadata.creation_time())
}

#[cfg(not(any(unix, windows)))]
fn log_identity(_metadata: &std::fs::Metadata) -> Option<u64> {
    None
}

#[cfg(test)]
#[path = "batch_tests.rs"]
mod batch_tests;
#[cfg(test)]
mod tests;
