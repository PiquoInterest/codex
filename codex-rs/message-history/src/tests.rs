use super::*;
use codex_config::types::History;
use pretty_assertions::assert_eq;
use std::fs::File;
use std::io::Write;
use tempfile::TempDir;

#[tokio::test]
async fn lookup_reads_history_entries() {
    let temp_dir = TempDir::new().expect("create temp dir");
    let history_path = temp_dir.path().join(HISTORY_FILENAME);

    let entries = vec![
        HistoryEntry {
            session_id: "first-session".to_string(),
            ts: 1,
            text: "first".to_string(),
        },
        HistoryEntry {
            session_id: "second-session".to_string(),
            ts: 2,
            text: "second".to_string(),
        },
    ];

    let mut file = File::create(&history_path).expect("create history file");
    for entry in &entries {
        writeln!(
            file,
            "{}",
            serde_json::to_string(entry).expect("serialize history entry")
        )
        .expect("write history entry");
    }

    let (log_id, count) = history_metadata_for_file(&history_path).await;
    assert_eq!(count, entries.len());

    let second_entry = lookup_history_entry(&history_path, log_id, /*offset*/ 1)
        .expect("fetch second history entry");
    assert_eq!(second_entry, entries[1]);
}

#[tokio::test]
async fn history_metadata_counts_newlines_across_read_boundaries() {
    let temp_dir = TempDir::new().expect("create temp dir");
    let history_path = temp_dir.path().join(HISTORY_FILENAME);
    let mut contents = vec![b'x'; 3 * HISTORY_COUNT_BUFFER_SIZE + 1];
    let newline_offsets = [
        0,
        HISTORY_COUNT_BUFFER_SIZE - 1,
        HISTORY_COUNT_BUFFER_SIZE,
        2 * HISTORY_COUNT_BUFFER_SIZE,
        contents.len() - 2,
    ];
    for offset in newline_offsets {
        contents[offset] = b'\n';
    }
    std::fs::write(&history_path, contents).expect("write history file");

    let (_, count) = history_metadata_for_file(&history_path).await;

    assert_eq!(count, newline_offsets.len());
}

#[tokio::test]
async fn lookup_uses_stable_log_id_after_appends() {
    let temp_dir = TempDir::new().expect("create temp dir");
    let history_path = temp_dir.path().join(HISTORY_FILENAME);

    let initial = HistoryEntry {
        session_id: "first-session".to_string(),
        ts: 1,
        text: "first".to_string(),
    };
    let appended = HistoryEntry {
        session_id: "second-session".to_string(),
        ts: 2,
        text: "second".to_string(),
    };

    let mut file = File::create(&history_path).expect("create history file");
    writeln!(
        file,
        "{}",
        serde_json::to_string(&initial).expect("serialize initial entry")
    )
    .expect("write initial entry");

    let (log_id, count) = history_metadata_for_file(&history_path).await;
    assert_eq!(count, 1);

    let mut append = std::fs::OpenOptions::new()
        .append(true)
        .open(&history_path)
        .expect("open history file for append");
    writeln!(
        append,
        "{}",
        serde_json::to_string(&appended).expect("serialize appended entry")
    )
    .expect("append history entry");

    let fetched = lookup_history_entry(&history_path, log_id, /*offset*/ 1)
        .expect("lookup appended history entry");
    assert_eq!(fetched, appended);
}

#[tokio::test]
async fn append_entry_caps_history_with_segment_rotation() {
    let codex_home = TempDir::new().expect("create temp dir");
    let mut history = History::default();
    let entry = "a".repeat(200);
    let config_probe = HistoryConfig::new(codex_home.path(), &history);
    append_entry(&entry, "conversation-id", &config_probe)
        .await
        .expect("write probe entry");
    let history_path = codex_home.path().join(HISTORY_FILENAME);
    let entry_len = std::fs::metadata(&history_path).expect("metadata").len();

    // Cap at roughly six entries; rotation triggers near a quarter of that.
    let max_bytes = usize::try_from(entry_len * 6).expect("cap fits usize");
    history.max_bytes = Some(max_bytes);
    let config = HistoryConfig::new(codex_home.path(), &history);

    for _ in 0..40 {
        append_entry(&entry, "conversation-id", &config)
            .await
            .expect("append entry");
    }

    let active_len = std::fs::metadata(&history_path).expect("metadata").len();
    assert!(
        active_len < entry_len * 3,
        "active file must stay below the rotation threshold plus one entry"
    );

    let set = SegmentSet::scan(codex_home.path());
    assert!(
        !set.segments.is_empty(),
        "rotation must have produced segments"
    );

    // Retained bytes stay bounded: segments within the cap (the newest may
    // overshoot by itself) plus a below-threshold active file.
    let total = set.total_bytes() + active_len;
    assert!(
        total <= (max_bytes as u64) + entry_len * 3,
        "retained bytes {total} exceed cap {max_bytes} plus rotation slack"
    );

    // The newest entry is always retrievable at the last global offset.
    let (log_id, count) = history_metadata(&config).await;
    assert!(count > 0);
    let newest = lookup(log_id, count - 1, &config).expect("newest entry resolves");
    assert_eq!(newest.text, entry);
}

#[tokio::test]
async fn append_entry_retains_newest_entry_under_tiny_cap() {
    let codex_home = TempDir::new().expect("create temp dir");
    let mut history = History::default();
    // Cap far below one entry: every append rotates immediately and deletes
    // every older segment, but the newest entry must always survive.
    history.max_bytes = Some(8);
    let config = HistoryConfig::new(codex_home.path(), &history);

    for index in 0..5 {
        let text = format!("entry-{index}");
        append_entry(&text, "conversation-id", &config)
            .await
            .expect("append entry");

        let (log_id, count) = history_metadata(&config).await;
        assert!(count > 0);
        let newest = lookup(log_id, count - 1, &config).expect("newest entry resolves");
        assert_eq!(newest.text, text);
    }
}

#[tokio::test]
async fn lookup_resolves_global_offsets_across_segments() {
    let codex_home = TempDir::new().expect("create temp dir");
    let mut history = History::default();
    let probe_config = HistoryConfig::new(codex_home.path(), &history);
    append_entry("probe", "conversation-id", &probe_config)
        .await
        .expect("write probe entry");
    let history_path = codex_home.path().join(HISTORY_FILENAME);
    let entry_len = std::fs::metadata(&history_path).expect("metadata").len();
    std::fs::remove_file(&history_path).expect("reset history");

    // Threshold of ~2.5 entries with a cap loose enough that nothing is
    // deleted: every entry ever appended must stay reachable via its global
    // offset even though the log spans multiple files.
    history.max_bytes = Some(usize::try_from(entry_len * 10).expect("cap fits usize"));
    let config = HistoryConfig::new(codex_home.path(), &history);

    let texts: Vec<String> = (0..8).map(|index| format!("entry-{index}")).collect();
    for text in &texts {
        // Pad to the probe length so rotation cadence is predictable.
        let padded = format!("{text:<width$}", width = "probe".len());
        append_entry(&padded, "conversation-id", &config)
            .await
            .expect("append entry");
    }

    let set = SegmentSet::scan(codex_home.path());
    assert!(
        !set.segments.is_empty(),
        "the corpus must span rotated segments for this test to bite"
    );

    let (log_id, count) = history_metadata(&config).await;
    assert_eq!(count, texts.len());
    for (offset, text) in texts.iter().enumerate() {
        let via_id = lookup(log_id, offset, &config).expect("entry resolves via active id");
        assert_eq!(via_id.text.trim_end(), text.as_str());
        let via_wildcard = lookup(0, offset, &config).expect("entry resolves via wildcard id");
        assert_eq!(via_wildcard.text.trim_end(), text.as_str());
    }
}

#[tokio::test]
async fn lookup_resolves_pre_rotation_identity_locally() {
    let codex_home = TempDir::new().expect("create temp dir");
    let mut history = History::default();
    let probe_config = HistoryConfig::new(codex_home.path(), &history);
    append_entry("first", "conversation-id", &probe_config)
        .await
        .expect("write first entry");
    let history_path = codex_home.path().join(HISTORY_FILENAME);
    let entry_len = std::fs::metadata(&history_path).expect("metadata").len();

    history.max_bytes = Some(usize::try_from(entry_len * 10).expect("cap fits usize"));
    let config = HistoryConfig::new(codex_home.path(), &history);

    append_entry("second", "conversation-id", &config)
        .await
        .expect("write second entry");
    let (pre_rotation_id, pre_count) = history_metadata(&config).await;
    assert_eq!(pre_count, 2);

    // Force at least one rotation so the pre-rotation file becomes a segment.
    for index in 0..6 {
        append_entry(&format!("later-{index}"), "conversation-id", &config)
            .await
            .expect("append entry");
    }
    let set = SegmentSet::scan(codex_home.path());
    assert!(!set.segments.is_empty(), "rotation must have happened");

    // Rotation renames the file in place, so the cached identity still names
    // it and the cached local offset still points at the same line.
    let entry = lookup(pre_rotation_id, 1, &config)
        .expect("pre-rotation identity resolves against the rotated segment");
    assert_eq!(entry.text, "second");
}

#[tokio::test]
async fn active_file_keeps_newest_entries_for_legacy_readers() {
    let codex_home = TempDir::new().expect("create temp dir");
    let mut history = History::default();
    let probe_config = HistoryConfig::new(codex_home.path(), &history);
    append_entry("probe", "conversation-id", &probe_config)
        .await
        .expect("write probe entry");
    let history_path = codex_home.path().join(HISTORY_FILENAME);
    let entry_len = std::fs::metadata(&history_path).expect("metadata").len();
    std::fs::remove_file(&history_path).expect("reset history");

    history.max_bytes = Some(usize::try_from(entry_len * 10).expect("cap fits usize"));
    let config = HistoryConfig::new(codex_home.path(), &history);
    for index in 0..8 {
        append_entry(&format!("entry-{index}"), "conversation-id", &config)
            .await
            .expect("append entry");
    }

    // A pre-segmentation binary reads only `history.jsonl`; it must see a
    // clean JSONL suffix of the logical history (possibly empty right after a
    // rotation), never a corrupt or unrelated file.
    let contents = std::fs::read_to_string(&history_path).expect("read active file");
    let mut seen = Vec::new();
    for line in contents.lines() {
        let entry: HistoryEntry = serde_json::from_str(line).expect("active line parses");
        seen.push(entry.text);
    }
    let (_, total) = history_metadata(&config).await;
    assert_eq!(total, 8);
    let expected_suffix: Vec<String> = (8 - seen.len()..8)
        .map(|index| format!("entry-{index}"))
        .collect();
    assert_eq!(seen, expected_suffix);
}

#[tokio::test]
async fn foreign_history_files_are_ignored() {
    let codex_home = TempDir::new().expect("create temp dir");
    std::fs::write(codex_home.path().join("history.abc.jsonl"), b"junk\n").expect("write junk");
    std::fs::write(codex_home.path().join("history..jsonl"), b"junk\n").expect("write junk");
    std::fs::write(codex_home.path().join("history.10x.jsonl"), b"junk\n").expect("write junk");

    let history = History::default();
    let config = HistoryConfig::new(codex_home.path(), &history);
    append_entry("only", "conversation-id", &config)
        .await
        .expect("append entry");

    let (log_id, count) = history_metadata(&config).await;
    assert_eq!(count, 1);
    let entry = lookup(log_id, 0, &config).expect("entry resolves");
    assert_eq!(entry.text, "only");
}

/// Manual write-amplification probe: appends a fixed corpus under a byte cap
/// and reports kernel-attributed write bytes. Run explicitly:
/// `cargo test --release -p codex-message-history -- --ignored --nocapture bench_append_storm`
#[tokio::test]
#[ignore = "manual benchmark, reports via --nocapture"]
async fn bench_append_storm_write_bytes() {
    use std::io::Read;
    fn proc_write_bytes() -> u64 {
        let mut s = String::new();
        std::fs::File::open("/proc/self/io")
            .expect("open /proc/self/io")
            .read_to_string(&mut s)
            .expect("read /proc/self/io");
        s.lines()
            .find_map(|l| l.strip_prefix("write_bytes: "))
            .expect("write_bytes field")
            .parse()
            .expect("parse write_bytes")
    }

    let n: usize = std::env::var("HISTBENCH_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(50_000);
    let cap: usize = std::env::var("HISTBENCH_CAP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1_048_576);

    let codex_home = TempDir::new().expect("create temp dir");
    let history = History {
        max_bytes: Some(cap),
        ..History::default()
    };
    let config = HistoryConfig::new(codex_home.path(), &history);
    let entry = "x".repeat(150);

    let start = std::time::Instant::now();
    let before = proc_write_bytes();
    for _ in 0..n {
        append_entry(&entry, "bench", &config)
            .await
            .expect("append entry");
    }
    let after = proc_write_bytes();

    let retained: u64 = std::fs::read_dir(codex_home.path())
        .expect("read dir")
        .flatten()
        .filter_map(|e| e.metadata().ok())
        .filter(std::fs::Metadata::is_file)
        .map(|m| m.len())
        .sum();
    let files = std::fs::read_dir(codex_home.path())
        .expect("read dir")
        .count();
    println!(
        "bench_append_storm: appends={n} cap={cap} write_bytes={} elapsed_ms={} retained_bytes={retained} files={files}",
        after - before,
        start.elapsed().as_millis(),
    );
}
