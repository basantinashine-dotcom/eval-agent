"""Check a written target before trusting it.

A target file (usually evals/target.py, plus evals/fakes.py) tells the runner
how to run someone's agent with pretend tools. Before anything runs, check:

1. Static: the written files don't run commands, touch the network, or change
   files (they only need to read and change the pretend world).
2. In a separate process, with the network blocked:
   - the target loads;
   - every tool server the agent uses has a pretend version, with exactly the
     same tool names, descriptions and input schemas as the real one;
   - each pretend tool runs once on an empty world and on the example world
     without crashing.
3. Nothing in the repo changed while checking.

Run on its own (used by check_target):  python -m eval_agent.check evals.target:make_target
"""

from __future__ import annotations

import ast
import asyncio
import importlib
import inspect
import json
import os
import subprocess
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Modules a target or fake file has no reason to import.
FORBIDDEN_MODULES = {
    "subprocess", "socket", "ssl", "requests", "httpx", "aiohttp", "urllib", "urllib3", "http", "ftplib",
    "smtplib", "imaplib", "poplib", "telnetlib", "webbrowser", "ctypes", "multiprocessing", "shutil",
    "paramiko", "boto3", "pty", "signal",
}
FORBIDDEN_NAMES = {"eval", "exec", "compile", "__import__", "breakpoint"}
# os.<these>() run commands or change files.
OS_ATTRS = {
    "system", "popen", "startfile", "spawnl", "spawnv", "spawnve", "execv", "execve", "execl", "fork",
    "kill", "remove", "unlink", "rmdir", "removedirs", "rename", "replace", "chmod", "chown", "putenv",
    "mkdir", "makedirs", "symlink",
}
# <anything>.<these>() change files (pathlib and friends).
FILE_ATTRS = {"unlink", "rmdir", "rmtree", "write_text", "write_bytes", "touch", "symlink_to", "mkdir"}

# Never compare these folders for changes: caches and the output folder itself.
_SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "env", "node_modules", ".mypy_cache", ".pytest_cache",
              ".ruff_cache", ".tox", "dist", "build", "runs"}


@dataclass
class ToolInfo:
    server: str
    name: str
    description: str = ""
    side_effect: bool = False


@dataclass
class CheckResult:
    ok: bool = False
    kind: str = ""                 # "sdk" or "custom"
    name: str = ""
    problems: list[str] = field(default_factory=list)   # must be fixed
    warnings: list[str] = field(default_factory=list)   # worth knowing
    tools: list[ToolInfo] = field(default_factory=list)
    real_servers: list[str] = field(default_factory=list)

    def text(self) -> str:
        lines = [("PASSED" if self.ok else "FAILED") + (f": {self.name} ({self.kind})" if self.name else "")]
        lines += [f"Problem: {p}" for p in self.problems]
        lines += [f"Warning: {w}" for w in self.warnings]
        if self.tools:
            lines.append("Pretend tools: " + ", ".join(
                f"{t.server}/{t.name}{' (changes things)' if t.side_effect else ''}" for t in self.tools))
        return "\n".join(lines)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckResult:
        data = dict(data)
        data["tools"] = [ToolInfo(**t) for t in data.get("tools", [])]
        return cls(**data)


def static_problems(paths: list[Path], root: Path) -> list[str]:
    """Things a target or fake file shouldn't do, found without running it."""
    problems = []
    root = root.resolve()
    for path in paths:
        path = path.resolve()
        rel = path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        except SyntaxError as exc:
            problems.append(f"{rel} line {exc.lineno}: not valid Python ({exc.msg})")
            continue
        for node in ast.walk(tree):
            where = f"{rel} line {getattr(node, 'lineno', '?')}"
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in FORBIDDEN_MODULES:
                        problems.append(f"{where}: imports {alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                if node.module.split(".")[0] in FORBIDDEN_MODULES:
                    problems.append(f"{where}: imports from {node.module}")
                elif node.module == "os" and any(a.name in OS_ATTRS for a in node.names):
                    problems.append(f"{where}: imports {', '.join(a.name for a in node.names)} from os")
            elif isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in FORBIDDEN_NAMES:
                    problems.append(f"{where}: calls {func.id}()")
                elif isinstance(func, ast.Attribute) and (
                        func.attr in FILE_ATTRS
                        or (func.attr in OS_ATTRS and isinstance(func.value, ast.Name) and func.value.id == "os")):
                    problems.append(f"{where}: calls .{func.attr}()")
                elif isinstance(func, ast.Name) and func.id == "open" and _opens_for_writing(node):
                    problems.append(f"{where}: opens a file for writing")
    return [f"{p} (pretend tools should only read and change the pretend world)" for p in problems]


def _opens_for_writing(call: ast.Call) -> bool:
    mode = call.args[1] if len(call.args) > 1 else next((k.value for k in call.keywords if k.arg == "mode"), None)
    if mode is None:
        return False
    if isinstance(mode, ast.Constant) and isinstance(mode.value, str):
        return any(ch in mode.value for ch in "wax+")
    return True  # a mode we can't read; assume the worst


def snapshot(root: Path, skip: set[str] | None = None) -> dict[str, tuple[int, int]]:
    """Size and modification time of every file under root, to spot changes."""
    skip = _SKIP_DIRS | (skip or set())
    files = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip]
        for name in filenames:
            path = Path(dirpath) / name
            try:
                st = path.stat()
            except OSError:
                continue
            files[path.relative_to(root).as_posix()] = (st.st_size, st.st_mtime_ns)
    return files


def changed_files(before: dict[str, tuple[int, int]], after: dict[str, tuple[int, int]]) -> list[str]:
    changed = [f"{p} (changed)" for p in before if p in after and before[p] != after[p]]
    changed += [f"{p} (new)" for p in after if p not in before]
    changed += [f"{p} (deleted)" for p in before if p not in after]
    return sorted(changed)


def check_target(spec: str, root: Path, written: list[Path] | None = None, *, timeout_s: float = 120,
                 skip_dirs: set[str] | None = None) -> CheckResult:
    """Check `module.path:function` from the repo at root. Never imports it into this process."""
    root = root.resolve()
    if written:
        problems = static_problems(written, root)
        if problems:
            return CheckResult(problems=problems)
    before = snapshot(root, skip_dirs)
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8",
           "PYTHONPATH": os.pathsep.join([str(root), os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)}
    try:
        out = subprocess.run([sys.executable, "-m", "eval_agent.check", spec], cwd=root, env=env,
                             capture_output=True, text=True, encoding="utf-8", timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return CheckResult(problems=[f"Loading the target took more than {timeout_s:.0f} seconds."])
    result = None
    for line in reversed(out.stdout.splitlines()):
        if line.startswith("{"):
            try:
                result = CheckResult.from_dict(json.loads(line))
                break
            except (json.JSONDecodeError, TypeError):
                pass
    if result is None:
        tail = (out.stderr or out.stdout).strip().splitlines()[-8:]
        result = CheckResult(problems=["The check itself couldn't finish:\n" + "\n".join(tail)])
    changes = changed_files(before, snapshot(root, skip_dirs))
    if changes:
        result.problems.append("Checking changed files in the repo, so something calls the real code: "
                               + ", ".join(changes[:10]))
        result.ok = False
    return result


# ---------------------------------------------------------------------------
# Everything below runs inside the separate process.

_LOOPBACK = {"127.0.0.1", "::1", "localhost"}


def _block_network() -> None:
    """Refuse connections to anything but this computer (asyncio needs loopback on Windows)."""
    import socket

    connect, connect_ex, getaddrinfo = socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo

    def host_of(address: Any) -> str:
        return str(address[0]) if isinstance(address, tuple) and address else str(address)

    def guard(host: str) -> None:
        if host not in _LOOPBACK:
            raise PermissionError(f"The network is blocked while checking a target (tried {host}).")

    def safe_connect(self: Any, address: Any) -> Any:
        guard(host_of(address))
        return connect(self, address)

    def safe_connect_ex(self: Any, address: Any) -> Any:
        guard(host_of(address))
        return connect_ex(self, address)

    def safe_getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        guard(str(host))
        return getaddrinfo(host, *args, **kwargs)

    socket.socket.connect = safe_connect  # type: ignore[method-assign]
    socket.socket.connect_ex = safe_connect_ex  # type: ignore[method-assign]
    socket.getaddrinfo = safe_getaddrinfo  # type: ignore[assignment]


def sample_args(schema: dict[str, Any]) -> dict[str, Any]:
    """Simple made-up arguments that fit a tool's input schema."""
    if not isinstance(schema, dict):
        return {}
    if "properties" not in schema:  # the SDK's short form: {"order_id": str}
        schema = {"properties": {k: {"type": _short_type(v)} for k, v in schema.items()}}
    args = {}
    for name, prop in (schema.get("properties") or {}).items():
        args[name] = _sample_value(name, prop if isinstance(prop, dict) else {})
    return args


def _short_type(py_type: Any) -> str:
    return {str: "string", int: "integer", float: "number", bool: "boolean", list: "array",
            dict: "object"}.get(py_type, "string")


def _sample_value(name: str, prop: dict[str, Any]) -> Any:
    if prop.get("enum"):
        return prop["enum"][0]
    kind = prop.get("type", "string")
    if isinstance(kind, list):
        kind = next((k for k in kind if k != "null"), "string")
    if kind == "integer":
        return 1
    if kind == "number":
        return 1.0
    if kind == "boolean":
        return True
    if kind == "array":
        return []
    if kind == "object":
        return {}
    return "test@example.com" if "email" in name.lower() else "test-1"


@asynccontextmanager
async def mcp_client(server: Any) -> AsyncIterator[Any]:
    """A real MCP client session connected to an in-process server.

    Uses only what mcp 1.x and 2.x share (the in-memory streams and Server.run,
    as the Agent SDK does), never the server's internals.
    """
    import anyio
    from mcp import ClientSession
    from mcp.shared.memory import create_client_server_memory_streams

    async with create_client_server_memory_streams() as (client_streams, server_streams):
        async with anyio.create_task_group() as tasks:
            async def serve() -> None:
                await server.run(server_streams[0], server_streams[1], server.create_initialization_options(),
                                 raise_exceptions=False)

            tasks.start_soon(serve)
            try:
                async with ClientSession(client_streams[0], client_streams[1]) as session:
                    await session.initialize()
                    yield session
            finally:
                tasks.cancel_scope.cancel()


def _field(obj: Any, *names: str) -> Any:
    """Read a field whichever spelling this mcp version uses (inputSchema or input_schema)."""
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


async def call_server_tool(server: Any, name: str, args: dict[str, Any]) -> tuple[str, bool]:
    """Call a tool on an in-process MCP server. Returns (text, is_error)."""
    async with mcp_client(server) as session:
        result = await session.call_tool(name, args)
    text = "".join(getattr(item, "text", "") for item in result.content)
    return text, bool(_field(result, "isError", "is_error"))


async def _list_server_tools(config: Any) -> list[dict[str, Any]] | None:
    """Tools of an in-process SDK server, or None for an external server we won't start."""
    if not (isinstance(config, dict) and config.get("type") == "sdk"):
        return None
    async with mcp_client(config["instance"]) as session:
        listed = await session.list_tools()
    return [{"name": t.name, "description": t.description or "",
             "schema": _field(t, "inputSchema", "input_schema")} for t in listed.tools]


def _smoke(server: str, fakes: list[Any], worlds: list[dict[str, Any]], result: CheckResult) -> None:
    from .fakes import FakeWorld, Toolbox

    for world_state in worlds:
        box = Toolbox(fakes, FakeWorld(world_state))
        for fake in fakes:
            box.call(fake.name, sample_args(fake.input_schema))
        for crash in box.crashes:
            label = "the example world" if world_state else "an empty world"
            result.problems.append(f"Pretend tool {server}/{crash} (on {label}, with made-up arguments). "
                                   "Pretend tools should handle missing data, e.g. with world.state.get().")


def _check_sdk(target: Any, result: CheckResult, worlds: list[dict[str, Any]]) -> None:
    from .fakes import FakeWorld, Toolbox

    result.kind = "sdk"
    options = target.options_factory()
    servers = options.mcp_servers
    if not isinstance(servers, dict):
        result.problems.append("The agent loads its tool servers from a config file, which isn't supported yet.")
        return
    result.real_servers = sorted(servers)
    builtins = options.tools is None or bool(options.tools)
    if builtins:
        names = "all of them" if options.tools is None else ", ".join(map(str, options.tools))
        result.warnings.append(f"The agent can use Claude Code's built-in tools ({names}). The runner gives it an "
                               "empty temporary folder, but shell commands could still reach the network.")
        if options.cwd:
            result.problems.append(f"The agent has built-in tools and its working folder is set to {options.cwd}, "
                                   "so a run could change real files there.")
    if getattr(options, "agents", None):
        result.warnings.append("The agent has subagents; their tools need pretend versions too.")

    for name in servers:
        if name not in target.fakes and name not in target.real_servers_ok:
            result.problems.append(f"Tool server '{name}' has no pretend version.")
    for name in target.fakes:
        if name not in servers:
            result.problems.append(f"There's a pretend server '{name}', but the agent has no tool server by "
                                   f"that name (it has: {', '.join(servers) or 'none'}).")

    for name, factory in target.fakes.items():
        fakes = factory()
        for fake in fakes:
            result.tools.append(ToolInfo(name, fake.name, fake.description, fake.side_effect))
        _smoke(name, fakes, worlds, result)
        if name not in servers:
            continue
        real = asyncio.run(_list_server_tools(servers[name]))
        if real is None:
            result.warnings.append(f"'{name}' is an external tool server; I can't compare its real tools with the "
                                   "pretend ones without starting it.")
        else:
            fake_listed = asyncio.run(_list_server_tools(Toolbox(fakes, FakeWorld()).sdk_server(name))) or []
            real_by_name = {t["name"]: t for t in real}
            fake_by_name = {t["name"]: t for t in fake_listed}
            for tool_name in sorted(set(real_by_name) - set(fake_by_name)):
                result.problems.append(f"{name}/{tool_name} is a real tool with no pretend version.")
            for tool_name in sorted(set(fake_by_name) - set(real_by_name)):
                result.problems.append(f"{name}/{tool_name} is a pretend tool the real agent doesn't have.")
            for tool_name in sorted(set(real_by_name) & set(fake_by_name)):
                r, f = real_by_name[tool_name], fake_by_name[tool_name]
                if r["schema"] != f["schema"]:
                    result.problems.append(f"{name}/{tool_name}: the pretend input schema differs from the real one "
                                           f"(real {json.dumps(r['schema'])}, pretend {json.dumps(f['schema'])}).")
                if r["description"] != f["description"]:
                    result.problems.append(f"{name}/{tool_name}: the pretend description differs from the real one, "
                                           "which changes how the agent uses it. Use fake_like().")


def _check_custom(target: Any, result: CheckResult, worlds: list[dict[str, Any]]) -> None:
    result.kind = "custom"
    try:
        params = inspect.signature(target.fn).parameters
        if len(params) < 2:
            result.problems.append("The run function must take (case, toolbox).")
    except (TypeError, ValueError):
        pass
    fakes = target.fakes()
    for fake in fakes:
        result.tools.append(ToolInfo("tools", fake.name, fake.description, fake.side_effect))
    if not fakes:
        result.warnings.append("There are no pretend tools, so the eval can't see any tool calls.")
    if not target.instructions:
        result.warnings.append("No instructions given, so the analyzer won't know the agent's rules.")
    result.warnings.append("This is custom code: I can't tell, without running it, that every real tool call goes "
                           "to a pretend tool. The first run checks that nothing in the repo changed.")
    _smoke("tools", fakes, worlds, result)


def run_check(spec: str) -> CheckResult:
    from .runner import CallableTarget, SdkTarget

    result = CheckResult()
    module_name, _, attr = spec.partition(":")
    try:
        module = importlib.import_module(module_name)
        factory = getattr(module, attr or "make_target")
        target = factory()
    except ModuleNotFoundError as exc:
        result.problems.append(f"Couldn't load the target: {exc}. If that's a package your agent needs, install "
                               "it where eval-agent runs (pip install ...).")
        return result
    except Exception as exc:
        result.problems.append(f"Couldn't load the target: {type(exc).__name__}: {exc}")
        return result
    result.name = getattr(target, "name", "")
    example = getattr(target, "example_world", None) or {}
    if not isinstance(example, dict):
        result.problems.append("example_world must be a dict.")
        example = {}
    worlds = [{}] + ([example] if example else [])
    if not example:
        result.warnings.append("No example_world, so I only tried the pretend tools on an empty world.")
    try:
        if isinstance(target, SdkTarget):
            _check_sdk(target, result, worlds)
        elif isinstance(target, CallableTarget):
            _check_custom(target, result, worlds)
        else:
            result.problems.append(f"{spec} returned a {type(target).__name__}; it must return an SdkTarget or "
                                   "a CallableTarget.")
    except Exception as exc:
        result.problems.append(f"Checking the target failed: {type(exc).__name__}: {exc}")
    result.ok = not result.problems
    return result


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    _block_network()
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    result = run_check(argv[0] if argv else "evals.target:make_target")
    print(json.dumps(asdict(result), ensure_ascii=False))
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
