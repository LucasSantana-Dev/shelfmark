#!/usr/bin/env python3
"""AST symbol extraction for the call-graph tables.

extract(text, language) -> (defs, calls, imports), all file-less (the caller adds the file):
  defs:    (symbol_name, language, start_line, end_line)
  calls:   (caller, callee, line_in_caller)
  imports: (imported_module, imported_name)

Python uses stdlib `ast` (high fidelity). TS/shell return empty for now (Phase 4: regex).
"""
from __future__ import annotations
import ast
import re


def extract_python(text: str):
    defs: list = []
    calls: list = []
    imports: list = []
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return defs, calls, imports

    scope: list[str] = []  # enclosing def/class names, for caller attribution

    class V(ast.NodeVisitor):
        def _def(self, node):
            end = getattr(node, "end_lineno", None) or node.lineno
            defs.append((node.name, "python", node.lineno, end))
            scope.append(node.name)
            self.generic_visit(node)
            scope.pop()

        visit_FunctionDef = _def
        visit_AsyncFunctionDef = _def
        visit_ClassDef = _def

        def visit_Call(self, node):
            f = node.func
            callee = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else None)
            if callee:
                calls.append((scope[-1] if scope else "<module>", callee, node.lineno))
            self.generic_visit(node)

        def visit_Import(self, node):
            for a in node.names:
                imports.append((a.name, a.asname or a.name.split(".")[-1]))
            self.generic_visit(node)

        def visit_ImportFrom(self, node):
            mod = node.module or ""
            for a in node.names:
                imports.append((mod, a.name))
            self.generic_visit(node)

    V().visit(tree)
    return defs, calls, imports


# --- TS/JS + shell: regex extraction (defs + TS imports; calls are regex-based, precision-biased) ---
_TS_DEF = re.compile(
    r"^(?P<indent>[ \t]*)"
    r"(?:export\s+(?:default\s+)?)?(?:declare\s+)?(?:async\s+)?"
    r"(?:function\*?\s+(?P<fn>[A-Za-z_$][\w$]*)"
    r"|class\s+(?P<cls>[A-Za-z_$][\w$]*)"
    r"|(?:interface|type|enum)\s+(?P<ty>[A-Za-z_$][\w$]*)"
    r"|(?:const|let|var)\s+(?P<var>[A-Za-z_$][\w$]*)\s*[:=])",
    re.MULTILINE,
)
_TS_IMPORT = re.compile(
    r"""^\s*import\s+(?:(?P<names>[^;'"]+?)\s+from\s+)?['"](?P<mod>[^'"]+)['"]""",
    re.MULTILINE,
)
# TS/JS call detection: identifier(. Excludes JS keywords and definition-line calls.
# Regex calls are heuristic (precision-biased): captures simple invocations, misses chained calls.
_TS_CALL = re.compile(
    r"(?<![A-Za-z_$\w])([A-Za-z_$][\w$]*)\s*\(",
)
_TS_KEYWORDS = {
    "if", "for", "while", "switch", "catch", "return", "function", "class",
    "const", "let", "var", "import", "export", "typeof", "await", "async",
    "new", "do", "else", "try", "throw", "in", "of", "case", "delete",
    "void", "yield", "super", "this"
}
_SHELL_DEF = re.compile(
    r"^(?P<indent>[ \t]*)(?:function\s+)?(?P<name>[A-Za-z_][\w-]*)\s*\(\)\s*\{",
    re.MULTILINE,
)


def _line_at(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def extract_ts(text: str, language: str):
    defs: list = []
    calls: list = []
    imports: list = []

    # Extract definitions (unchanged).
    matches = [m for m in _TS_DEF.finditer(text) if not m.group("indent")]
    n_lines = text.count("\n") + 1
    def_ranges: list[tuple[str, int, int]] = []  # (name, start_line, end_line)
    for idx, m in enumerate(matches):
        name = m.group("fn") or m.group("cls") or m.group("ty") or m.group("var")
        if not name:
            continue
        start = _line_at(text, m.start())
        end = (_line_at(text, matches[idx + 1].start()) - 1) if idx + 1 < len(matches) else n_lines
        defs.append((name, language, start, max(start, end)))
        def_ranges.append((name, start, max(start, end)))

    # Extract imports (unchanged).
    for m in _TS_IMPORT.finditer(text):
        mod = m.group("mod")
        idents = re.findall(r"[A-Za-z_$][\w$]*", m.group("names") or "")
        for nm in (idents or [mod.rsplit("/", 1)[-1]]):
            if nm not in ("from", "as", "type", "default"):
                imports.append((mod, nm))

    # Extract calls: find identifier( patterns, exclude keywords, attribute to enclosing def.
    # Regex calls are heuristic (precision-biased): captures simple invocations, misses chained calls.
    # Build a set of line numbers where defs start to skip the def-keyword calls.
    def_lines: set = set()
    for m in _TS_DEF.finditer(text):
        def_line = _line_at(text, m.start())
        def_lines.add(def_line)

    seen_calls: set = set()  # De-dup (caller, callee, line)

    for m in _TS_CALL.finditer(text):
        callee = m.group(1)
        # Skip keywords
        if callee in _TS_KEYWORDS:
            continue

        call_line = _line_at(text, m.start())

        # Skip calls that appear to be the def-keyword itself (e.g., "function f()")
        # by checking if the callee matches the name being defined on this line
        is_def_keyword = False
        for match in _TS_DEF.finditer(text):
            if _line_at(text, match.start()) == call_line:
                def_name = match.group("fn") or match.group("cls") or match.group("ty") or match.group("var")
                if def_name == callee:
                    is_def_keyword = True
                    break
        if is_def_keyword:
            continue

        call_line = _line_at(text, m.start())

        # Find enclosing def by line range
        caller = "<module>"
        for name, start, end in def_ranges:
            if start <= call_line <= end:
                caller = name
                break

        call_tuple = (caller, callee, call_line)
        if call_tuple not in seen_calls:
            calls.append(call_tuple)
            seen_calls.add(call_tuple)

    return defs, calls, imports


def extract_shell(text: str):
    defs: list = []
    calls: list = []
    lines = text.splitlines()

    # Extract definitions.
    def_names: set = set()
    def_ranges: list[tuple[str, int, int]] = []  # (name, start_line, end_line)
    for m in _SHELL_DEF.finditer(text):
        if m.group("indent"):
            continue
        start = _line_at(text, m.start())
        depth, i, end = 0, start - 1, start
        while i < len(lines):
            depth += lines[i].count("{") - lines[i].count("}")
            i += 1
            if depth <= 0 and i > start - 1:
                end = i
                break
        name = m.group("name")
        defs.append((name, "shell", start, max(start, end)))
        def_names.add(name)
        def_ranges.append((name, start, max(start, end)))

    # Conservative call extraction: only emit calls to functions DEFINED in the same file.
    # Regex calls are heuristic (precision-biased): unreliable for shell, so require intra-file definition.
    # Pattern: name(  with word-boundary start, where name is in def_names.
    if def_names:
        # Build regex for intra-file function calls
        safe_names = "|".join(re.escape(name) for name in def_names)
        call_pattern = re.compile(rf"\b({safe_names})\s*\(")
        seen_calls: set = set()

        for m in call_pattern.finditer(text):
            callee = m.group(1)
            call_line = _line_at(text, m.start())

            # Find enclosing def
            caller = "<module>"
            for name, start, end in def_ranges:
                if start <= call_line <= end:
                    caller = name
                    break

            call_tuple = (caller, callee, call_line)
            if call_tuple not in seen_calls:
                calls.append(call_tuple)
                seen_calls.add(call_tuple)

    return defs, calls, []


def extract(text: str, language: str):
    if language == "python":
        return extract_python(text)
    if language in ("typescript", "javascript"):
        return extract_ts(text, language)
    if language == "shell":
        return extract_shell(text)
    return [], [], []
