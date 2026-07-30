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
const HISTORY_FILENAME: &str = "history.jsonl";
const HISTORY_READ_BUFFER_SIZE: usize = 8192;
/// Larger buffer for the whole-file newline count at thread open; the batch
/// scanner keeps the smaller buffer, which its chunk-stitching tests and
/// stack-allocated read path are sized around.
const HISTORY_COUNT_BUFFER_SIZE: usize = 256 * 1024;

/// When history exceeds the hard cap, trim it down to this fraction of `max_bytes`.
const HISTORY_SOFT_CAP_RATIO: f64 = 0.8;

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

    let mut history_file = options.open(&path)?;

    // Ensure permissions.
    ensure_owner_only_permissions(&history_file).await?;

    let history_max_bytes = config.max_bytes;

    // Perform a blocking write under an advisory write lock using std::fs.
    tokio::task::spawn_blocking(move || -> Result<()> {
        // Retry a few times to avoid indefinite blocking when contended.
        for _ in 0..MAX_RETRIES {
            match history_file.try_lock() {
                Ok(()) => {
                    // While holding the exclusive lock, write the full line.
                    // We do not open the file with `append(true)` on Windows, so ensure the
                    // cursor is positioned at the end before writing.
                    history_file.seek(SeekFrom::End(0))?;
                    history_file.write_all(line.as_bytes())?;
                    history_file.flush()?;
                    enforce_history_limit(&mut history_file, history_max_bytes)?;
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

/// Trim the history file to honor `max_bytes`, dropping the oldest lines while holding
/// the write lock so the newest entry is always retained. When the file exceeds the
/// hard cap, it rewrites the remaining tail to a soft cap to avoid trimming again
/// immediately on the next write.
fn enforce_history_limit(file: &mut File, max_bytes: Option<usize>) -> Result<()> {
    let Some(max_bytes) = max_bytes else {
        return Ok(());
    };

    if max_bytes == 0 {
        return Ok(());
    }

    let max_bytes = match u64::try_from(max_bytes) {
        Ok(value) => value,
        Err(_) => return Ok(()),
    };

    let current_len = file.metadata()?.len();

    if current_len <= max_bytes {
        return Ok(());
    }

    let mut reader = file.try_clone()?;

    // Locate the newest entry with a bounded backward scan instead of reading
    // the whole file forward line by line; only the newest entry's length
    // matters for the trim target.
    let newest_start = last_line_start(&mut reader, current_len)?;
    let trim_target = trim_target_bytes(max_bytes, current_len - newest_start);
    let excess = current_len.saturating_sub(trim_target);
    if excess == 0 || newest_start == 0 {
        return Ok(());
    }

    // Scan newline boundaries forward over only the oldest entries, stopping
    // at the first boundary that discards at least `excess` bytes. The newest
    // entry is never dropped; when every older entry is too small the whole
    // older prefix goes. Boundaries are byte-defined, so unlike the previous
    // whole-file `read_line` pass this does not validate (or fail on) invalid
    // UTF-8 in entries that are about to be discarded.
    let mut drop_bytes = newest_start;
    reader.seek(SeekFrom::Start(0))?;
    let mut buf = vec![0u8; HISTORY_READ_BUFFER_SIZE];
    let mut pos = 0u64;
    'scan: while pos < newest_start {
        let want = usize::try_from((newest_start - pos).min(HISTORY_READ_BUFFER_SIZE as u64))
            .unwrap_or(HISTORY_READ_BUFFER_SIZE);
        let read = reader.read(&mut buf[..want])?;
        if read == 0 {
            break;
        }
        for offset in memchr_iter(b'\n', &buf[..read]) {
            let line_end = pos + offset as u64 + 1;
            if line_end >= excess {
                drop_bytes = line_end;
                break 'scan;
            }
        }
        pos += read as u64;
    }

    if drop_bytes == 0 {
        return Ok(());
    }

    reader.seek(SeekFrom::Start(drop_bytes))?;

    let capacity = usize::try_from(current_len.saturating_sub(drop_bytes)).unwrap_or(0);
    let mut tail = Vec::with_capacity(capacity);

    reader.read_to_end(&mut tail)?;

    file.set_len(0)?;
    file.seek(SeekFrom::Start(0))?;
    file.write_all(&tail)?;
    file.flush()?;

    Ok(())
}

/// Returns the byte offset where the newest entry begins, scanning backward in
/// bounded chunks. A single trailing newline terminates the newest entry
/// rather than starting an empty one, matching `read_line` semantics.
fn last_line_start(file: &mut File, len: u64) -> Result<u64> {
    if len == 0 {
        return Ok(0);
    }
    let mut buf = vec![0u8; HISTORY_READ_BUFFER_SIZE];
    let mut end = len;
    let mut first_chunk = true;
    while end > 0 {
        let start = end.saturating_sub(HISTORY_READ_BUFFER_SIZE as u64);
        let chunk_len = usize::try_from(end - start).unwrap_or(HISTORY_READ_BUFFER_SIZE);
        file.seek(SeekFrom::Start(start))?;
        file.read_exact(&mut buf[..chunk_len])?;
        let mut search_end = chunk_len;
        if first_chunk {
            if buf[chunk_len - 1] == b'\n' {
                search_end -= 1;
            }
            first_chunk = false;
        }
        if let Some(idx) = memchr::memrchr(b'\n', &buf[..search_end]) {
            return Ok(start + idx as u64 + 1);
        }
        end = start;
    }
    Ok(0)
}

fn trim_target_bytes(max_bytes: u64, newest_entry_len: u64) -> u64 {
    let soft_cap_bytes = ((max_bytes as f64) * HISTORY_SOFT_CAP_RATIO)
        .floor()
        .clamp(1.0, max_bytes as f64) as u64;

    soft_cap_bytes.max(newest_entry_len)
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
    history_metadata_for_file(&path).await
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
    use std::io::BufRead;
    use std::io::BufReader;

    let file: File = match OpenOptions::new().read(true).open(path) {
        Ok(f) => f,
        Err(e) => {
            tracing::warn!(error = %e, "failed to open history file");
            return None;
        }
    };

    let metadata = match file.metadata() {
        Ok(m) => m,
        Err(e) => {
            tracing::warn!(error = %e, "failed to stat history file");
            return None;
        }
    };

    let current_log_id = log_identity(&metadata)?;

    if log_id != 0 && current_log_id != log_id {
        return None;
    }

    // Open & lock file for reading using a shared lock.
    // Retry a few times to avoid indefinite blocking.
    for _ in 0..MAX_RETRIES {
        let lock_result = file.try_lock_shared();

        match lock_result {
            Ok(()) => {
                let reader = BufReader::new(&file);
                for (idx, line_res) in reader.lines().enumerate() {
                    let line = match line_res {
                        Ok(l) => l,
                        Err(e) => {
                            tracing::warn!(error = %e, "failed to read line from history file");
                            return None;
                        }
                    };

                    if idx == offset {
                        match serde_json::from_str::<HistoryEntry>(&line) {
                            Ok(entry) => return Some(entry),
                            Err(e) => {
                                tracing::warn!(error = %e, "failed to parse history entry");
                                return None;
                            }
                        }
                    }
                }
                // Not found at requested offset.
                return None;
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
