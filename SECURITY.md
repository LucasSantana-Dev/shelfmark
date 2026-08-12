# Security Policy

## Supported versions

Only the latest release (and `main`) receives security fixes.

| Version | Supported |
|---------|-----------|
| latest release / `main` | yes |
| older releases | no |

## Reporting a vulnerability

Please do **not** open a public issue for security problems. Instead, report
privately via GitHub's private vulnerability reporting:

**[Report a vulnerability](https://github.com/LucasSantana-Dev/shelfmark/security/advisories/new)**

Include what you can: affected file/function, reproduction steps or PoC, and
impact. You should get an initial response within 7 days. Once a fix ships,
the advisory is published and you are credited (unless you prefer not to be).

## Scope notes

shelfmark is a local-first tool: it runs on your machine, indexes files you
point it at, and makes no network calls except the one-time embedding model
download from Hugging Face. Things that are in scope:

- Path traversal or writes outside `RAG_HOME` during indexing/packing
- Code execution triggered by indexing a malicious file or `sources.yaml`
- MCP server issues (`mcp_server.py`): injection via tool arguments,
  responses leaking files outside configured sources
- Unsafe deserialization (index/cache loading)

Out of scope: vulnerabilities in dependencies themselves (report upstream;
a report here is still welcome if shelfmark uses the dependency unsafely),
and issues requiring an already-compromised machine.
