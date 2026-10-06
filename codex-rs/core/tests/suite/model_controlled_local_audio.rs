use codex_features::Feature;
use codex_protocol::ThreadId;
use codex_protocol::models::PermissionProfile;
use codex_protocol::openai_models::InputModality;
use codex_protocol::permissions::FileSystemAccessMode;
use codex_protocol::permissions::FileSystemPath;
use codex_protocol::permissions::FileSystemSandboxEntry;
use codex_protocol::permissions::FileSystemSandboxPolicy;
use codex_protocol::permissions::NetworkSandboxPolicy;
use codex_protocol::protocol::AskForApproval;
use codex_protocol::protocol::EventMsg;
use codex_utils_absolute_path::AbsolutePathBuf;
use core_test_support::responses::ev_assistant_message;
use core_test_support::responses::ev_completed;
use core_test_support::responses::ev_function_call_with_namespace;
use core_test_support::responses::ev_response_created;
use core_test_support::responses::mount_sse_once_match;
use core_test_support::responses::sse;
use core_test_support::responses::start_mock_server;
use core_test_support::test_codex::test_codex;
use core_test_support::wait_for_event_with_timeout;
use pretty_assertions::assert_eq;
use serde_json::json;
use std::fs;
use std::time::Duration;

const TEST_MODEL: &str = "gpt-5.2";
const PARENT_PROMPT: &str = "inspect the repository";
const CHILD_PROMPT: &str = "child: inspect repository";
const SPAWN_CALL_ID: &str = "spawn-local-audio";
const SECRET_BYTES: &[u8] = b"secret";
const AUDIO_DATA_URL: &str = "data:audio/wav;base64,c2VjcmV0";

fn body_contains(request: &wiremock::Request, text: &str) -> bool {
    let is_zstd = request
        .headers
        .get("content-encoding")
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| {
            value
                .split(',')
                .any(|entry| entry.trim().eq_ignore_ascii_case("zstd"))
        });
    let body = if is_zstd {
        zstd::stream::decode_all(std::io::Cursor::new(&request.body)).ok()
    } else {
        Some(request.body.clone())
    };
    body.and_then(|body| String::from_utf8(body).ok())
        .is_some_and(|body| body.contains(text))
}

#[cfg(unix)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn v1_spawn_agent_rejects_local_audio_symlink_denied_by_permission_profile()
-> anyhow::Result<()> {
    let server = start_mock_server().await;
    let mut builder = test_codex()
        .with_config(|config| {
            config
                .features
                .enable(Feature::Collab)
                .expect("test config should allow collaboration");
        })
        .with_model_info_override(TEST_MODEL, |model_info| {
            model_info.input_modalities.push(InputModality::Audio);
        });
    let test = builder.build_with_auto_env(&server).await?;

    // The session's actual working directory is the untrusted repository.
    let repository = test.config.cwd.as_path().to_path_buf();
    fs::create_dir_all(&repository)?;
    fs::write(repository.join("README.md"), b"test repository")?;

    let host_dir = tempfile::tempdir()?;
    let host_file = host_dir.path().join("host-secret.wav");
    fs::write(&host_file, SECRET_BYTES)?;
    let audio_path = repository.join("leak.wav");
    std::os::unix::fs::symlink(&host_file, &audio_path)?;

    let resolved_repository = fs::canonicalize(&repository)?;
    let resolved_audio_path = fs::canonicalize(&audio_path)?;
    let resolved_host_file = fs::canonicalize(&host_file)?;
    assert_eq!(resolved_audio_path, resolved_host_file);
    assert!(!resolved_audio_path.starts_with(&resolved_repository));

    let repository_root = AbsolutePathBuf::from_absolute_path(&resolved_repository)?;
    let denied_target = AbsolutePathBuf::from_absolute_path(&resolved_host_file)?;
    let file_system_policy = FileSystemSandboxPolicy::restricted(vec![
        FileSystemSandboxEntry::new(
            FileSystemPath::Path {
                path: repository_root.into(),
            },
            FileSystemAccessMode::Read,
        ),
        FileSystemSandboxEntry::new(
            FileSystemPath::Path {
                path: denied_target.into(),
            },
            FileSystemAccessMode::Deny,
        ),
    ]);
    // Lexical checking alone sees a path inside the repository. The resolved
    // symlink target is outside it and is explicitly denied by the profile.
    assert!(file_system_policy.can_read_local_path_with_cwd(&audio_path, &resolved_repository));
    assert!(
        !file_system_policy
            .can_read_local_path_with_cwd(&resolved_audio_path, &resolved_repository)
    );

    let permission_profile = PermissionProfile::from_runtime_permissions(
        &file_system_policy,
        NetworkSandboxPolicy::Restricted,
    );

    let spawn_args = serde_json::to_string(&json!({
        "items": [
            { "type": "text", "text": CHILD_PROMPT },
            { "type": "local_audio", "path": audio_path.display().to_string() },
        ],
    }))?;

    let parent_initial_request = mount_sse_once_match(
        &server,
        |request: &wiremock::Request| body_contains(request, PARENT_PROMPT),
        sse(vec![
            ev_response_created("resp-parent-1"),
            ev_function_call_with_namespace(
                SPAWN_CALL_ID,
                "multi_agent_v1",
                "spawn_agent",
                &spawn_args,
            ),
            ev_completed("resp-parent-1"),
        ]),
    )
    .await;

    let _child_request = mount_sse_once_match(
        &server,
        |request: &wiremock::Request| {
            body_contains(request, CHILD_PROMPT) && body_contains(request, AUDIO_DATA_URL)
        },
        sse(vec![
            ev_response_created("resp-child-1"),
            ev_assistant_message("msg-child-1", "child done"),
            ev_completed("resp-child-1"),
        ]),
    )
    .await;

    let parent_followup_request = mount_sse_once_match(
        &server,
        |request: &wiremock::Request| body_contains(request, SPAWN_CALL_ID),
        sse(vec![
            ev_response_created("resp-parent-2"),
            ev_assistant_message("msg-parent-2", "done"),
            ev_completed("resp-parent-2"),
        ]),
    )
    .await;

    test.submit_turn_with_approval_and_permission_profile(
        PARENT_PROMPT,
        AskForApproval::OnRequest,
        permission_profile,
    )
    .await?;

    assert_eq!(parent_initial_request.requests().len(), 1);
    assert_eq!(parent_followup_request.requests().len(), 1);

    let followup = parent_followup_request
        .last_request()
        .expect("parent should receive the rejected tool result");
    let tool_output = followup
        .function_call_output_text(SPAWN_CALL_ID)
        .expect("parent follow-up should include the spawn_agent result");
    if let Ok(tool_result) = serde_json::from_str::<serde_json::Value>(&tool_output)
        && let Some(agent_id) = tool_result
            .get("agent_id")
            .and_then(serde_json::Value::as_str)
    {
        let child_thread_id = ThreadId::from_string(agent_id)?;
        let child_thread = test.thread_manager.get_thread(child_thread_id).await?;
        wait_for_event_with_timeout(
            &child_thread,
            |event| matches!(event, EventMsg::TurnComplete(_) | EventMsg::Error(_)),
            Duration::from_secs(5),
        )
        .await;
    }
    let requests = server
        .received_requests()
        .await
        .expect("mock server should return captured requests");
    assert!(
        requests
            .iter()
            .all(|request| !body_contains(request, AUDIO_DATA_URL)),
        "denied audio data appeared in an outbound request"
    );
    assert!(
        requests
            .iter()
            .all(|request| !body_contains(request, "c2VjcmV0")),
        "base64-encoded denied file contents appeared in an outbound request"
    );
    assert!(
        requests.iter().all(|request| {
            !body_contains(request, CHILD_PROMPT) || !body_contains(request, AUDIO_DATA_URL)
        }),
        "a child request containing the denied audio was sent"
    );
    let response_request_count = requests
        .iter()
        .filter(|request| request.url.path().ends_with("/responses"))
        .count();
    assert_eq!(
        response_request_count, 2,
        "only the parent request and its follow-up should be sent"
    );
    assert!(
        tool_output.contains("Local media file access is denied"),
        "unexpected tool output: {tool_output}"
    );

    Ok(())
}
