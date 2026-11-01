#!/usr/bin/env python3
"""Tests for capture.py: hook payloads from each harness are normalized and
sanitized, automatic handoffs are claimed exactly once by another session,
written handoffs win, client layers stay apart, and purge drops the memory."""

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

TMP = Path(tempfile.mkdtemp(prefix="sqlrecall-capture-")).resolve()
os.environ["RAG_HOME"] = str(TMP)
os.environ["RAG_SOURCES"] = str(TMP / "sources.yaml")
os.environ.pop("RAG_CLIENT", None)

PROJ, ACME, GONE = TMP / "proj", TMP / "acme", TMP / "gone"
for d in (PROJ / ".git", ACME / "app" / ".git", GONE):
    d.mkdir(parents=True)
(TMP / "sources.yaml").write_text(textwrap.dedent(f"""\
    sources:
      - {{type: memory, glob: "{TMP.as_posix()}/notes/**/*.md"}}
    clients:
      acme:
        roots: ["{ACME.as_posix()}"]
    """))
# A client purged earlier: its tombstone outlives its sources.yaml entry.
(TMP / "index.client-gone.purged").write_text(json.dumps({"client": "gone", "roots": [str(GONE)]}))

sys.path.insert(0, str(Path(__file__).resolve().parent))
import capture  # noqa: E402
import config  # noqa: E402

GENERAL = capture.memory_path(config.DB)
CLIENT = capture.memory_path(config.client_db("acme"))


def run(agent, event, payload):
    return capture.handle(agent, event, payload)


def session(agent, sid, cwd, prompt="fix the flaky login test", reply="Fixed it; next run the e2e suite."):
    run(agent, "session-start", {"session_id": sid, "cwd": str(cwd)})
    run(agent, "user-prompt", {"session_id": sid, "cwd": str(cwd), "prompt": prompt})
    run(agent, "post-tool", {"session_id": sid, "cwd": str(cwd), "tool_name": "Edit",
                             "tool_input": {"file_path": "src/login.ts"}, "tool_response": {"success": True}})
    run(agent, "post-tool", {"session_id": sid, "cwd": str(cwd), "tool_name": "Bash",
                             "tool_input": {"command": "npm test"}, "tool_response": {"exit_code": 1, "stdout": "1 failed"}})
    run(agent, "stop", {"session_id": sid, "cwd": str(cwd), "last_assistant_message": reply})


def rows(db, sql, *args):
    con = sqlite3.connect(db)
    try:
        return con.execute(sql, args).fetchall()
    finally:
        con.close()


def test_sanitizer_redacts_secrets():
    # Built at runtime so no secret-shaped literal sits in the repo.
    gh = "ghp_" + "a1B2" * 9
    aws = "AKIA" + "ABCDEFGHIJKLMNOP"
    key = "sk-" + "x" * 30
    text = f"token={gh} aws {aws} Authorization: Bearer {'q' * 24} OPENAI_API_KEY='{key}' postgres://u:hunter22@db/x"
    out = capture.sanitize(text)
    for secret in (gh, aws, key, "q" * 24, "hunter22"):
        assert secret not in out, (secret, out)
    assert "[REDACTED" in out
    assert capture.sanitize("x" * (capture.BODY_MAX + 10)).endswith("[truncated 10 chars]")


def test_normalize_each_harness():
    claude = capture.normalize({"session_id": "s1", "cwd": "/w", "tool_name": "Bash",
                                "tool_input": {"command": "ls"}, "tool_response": {"stdout": "a"}})
    codex = capture.normalize({"session_id": "s2", "cwd": "/w", "prompt": "hi", "last_assistant_message": "done"})
    opencode = capture.normalize({"sessionID": "s3", "directory": "/w", "tool": "edit",
                                  "args": {"filePath": "a.ts"}, "output": {"output": "ok"},
                                  "prompt": [{"type": "text", "text": "do it"}]})
    assert (claude["session_id"], claude["tool"], claude["tool_input"]["command"]) == ("s1", "Bash", "ls")
    assert (codex["prompt"], codex["last_message"]) == ("hi", "done")
    assert (opencode["session_id"], opencode["cwd"], opencode["tool"]) == ("s3", "/w", "edit")
    assert capture._target(opencode["tool_input"]) == "a.ts" and opencode["prompt"] == "do it"


def test_transcript_tail_reads_last_reply():
    t = TMP / "t.jsonl"
    t.write_text("\n".join(json.dumps(r) for r in [
        {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "old"}]}},
        {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "claude last"}]}},
        {"type": "user", "message": {"role": "user", "content": "thanks"}},
    ]))
    assert capture.last_reply_from_transcript(str(t)) == "claude last"
    t.write_text(json.dumps({"type": "response_item", "payload": {"type": "message", "role": "assistant",
                                                                  "content": [{"type": "output_text", "text": "codex last"}]}}))
    assert capture.last_reply_from_transcript(str(t)) == "codex last"


def test_auto_handoff_claimed_once_by_another_session():
    session("claude-code", "A", PROJ)
    session("claude-code", "A", PROJ, prompt="also update the docs")  # refreshed in place, not duplicated
    assert len(rows(GENERAL, "SELECT id FROM handoffs WHERE from_session='A' AND state='open'")) == 1
    assert run("claude-code", "session-start", {"session_id": "A", "cwd": str(PROJ)}) == ""  # resume: no self-claim
    got = run("codex", "session-start", {"session_id": "B", "cwd": str(PROJ / "sub")})
    assert got.startswith("<sqlrecall-handoff") and 'from="claude-code"' in got, got
    for part in ("also update the docs", "src/login.ts", "FAILED npm test", "next run the e2e suite"):
        assert part in got, (part, got)
    assert run("opencode", "session-start", {"sessionID": "C", "directory": str(PROJ)}) == ""
    assert rows(GENERAL, "SELECT claimed_by_agent, claimed_by_session FROM handoffs WHERE from_session='A'") == [("codex", "B")]


def test_written_handoff_beats_auto_and_supersedes_older_auto():
    os.environ["SQLRECALL_HANDOFF_ON_START"] = "off"  # E's own start would claim D otherwise
    session("claude-code", "D", PROJ, prompt="older work")
    session("claude-code", "E", PROJ, prompt="newer work")
    del os.environ["SQLRECALL_HANDOFF_ON_START"]
    con = capture.connect(GENERAL)
    try:
        capture.begin_handoff(con, str(PROJ), "codex", "mcp:1", "ship the release", "Tag v2 after CI is green.")
    finally:
        con.close()
    first = run("claude-code", "session-start", {"session_id": "F", "cwd": str(PROJ)})
    assert 'kind="written"' in first and "Tag v2" in first, first
    second = run("claude-code", "session-start", {"session_id": "G", "cwd": str(PROJ)})
    assert "newer work" in second, second
    assert rows(GENERAL, "SELECT state FROM handoffs WHERE from_session='D'") == [("expired",)]


def test_worktree_shares_project_with_main_checkout():
    wt = TMP / "wt"
    wt.mkdir()
    (PROJ / ".git" / "worktrees" / "wt").mkdir(parents=True)
    (wt / ".git").write_text(f"gitdir: {PROJ / '.git' / 'worktrees' / 'wt'}\n")
    assert capture.project_key(str(wt)) == capture.project_key(str(PROJ))


def test_client_layer_and_purged_roots():
    session("claude-code", "K", ACME / "app")
    assert rows(CLIENT, "SELECT count(*) FROM sessions WHERE id='K'") == [(1,)]
    assert rows(GENERAL, "SELECT count(*) FROM sessions WHERE id='K'") == [(0,)]
    session("claude-code", "P", GONE)
    assert rows(GENERAL, "SELECT count(*) FROM sessions WHERE id='P'") == [(0,)]
    import mcp_server

    try:
        mcp_server._handoff_call("handoff_list", {"cwd": str(ACME / "app")})
    except ValueError:
        pass
    else:
        raise AssertionError("general-layer MCP server read a client's handoffs")
    assert "No open" in mcp_server._handoff_call("handoff_list", {"cwd": str(PROJ)})


def test_hook_cli_never_fails():
    script = Path(__file__).resolve().parent / "capture.py"
    for stdin in ("not json", "", json.dumps({"session_id": "Z", "cwd": str(PROJ), "prompt": "hi"})):
        p = subprocess.run([sys.executable, str(script), "--agent", "claude-code", "--event", "user-prompt"],
                           input=stdin, capture_output=True, text=True, env=os.environ)
        assert p.returncode == 0, p.stderr


def test_purge_drops_client_memory():
    import purge

    # A session recorded in general before acme's roots existed in sources.yaml.
    con = capture.connect(GENERAL)
    try:
        con.execute("INSERT INTO sessions(id, agent, project, cwd, started_at) VALUES ('old','x','p',?,0)",
                    (str(ACME / "app"),))
        con.execute("INSERT INTO observations(session_id, kind, title, body, ts) VALUES ('old','prompt','t','b',0)")
    finally:
        con.close()
    assert purge.main(["acme", "--apply", "--no-archive"]) == 0
    assert not CLIENT.exists()
    assert rows(GENERAL, "SELECT count(*) FROM observations WHERE session_id='old'") == [(0,)]
    assert rows(GENERAL, "SELECT count(*) FROM sessions WHERE id='A'") == [(1,)]


if __name__ == "__main__":
    order = ["test_sanitizer_redacts_secrets", "test_normalize_each_harness", "test_transcript_tail_reads_last_reply",
             "test_auto_handoff_claimed_once_by_another_session",
             "test_written_handoff_beats_auto_and_supersedes_older_auto",
             "test_worktree_shares_project_with_main_checkout", "test_client_layer_and_purged_roots",
             "test_hook_cli_never_fails", "test_purge_drops_client_memory"]
    assert len(order) == len([k for k in globals() if k.startswith("test_")]), "update the ordered list"
    try:
        for name in order:
            globals()[name]()
            print(f"ok  {name}")
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print(f"\n{len(order)} passed")
