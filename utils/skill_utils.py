"""Discovery, validation, and Pi-style execution for local Skill packages."""

from __future__ import annotations

import codecs
import hashlib
import html
import json
import os
import re
import threading
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

from .config_utils import get_config_section
from .request_utils import FunctionToolsRejected, ToolChoiceRejected


SKILL_OPTIONS_TYPE = "ycyy.openai_text_skill_options"
SKILL_OPTIONS_SCHEMA_VERSION = 1
DEFAULT_LIMITS = {
    "max_skill_md_bytes": 128 * 1024,
    "max_reference_file_bytes": 256 * 1024,
    "max_reference_total_bytes": 8 * 1024 * 1024,
    "max_disclosed_bytes_per_execution": 512 * 1024,
    "max_tool_rounds": 8,
    "max_tool_calls_per_execution": 16,
}
DEFAULT_ALLOW_CALL = False
PLUGIN_ROOT = Path(__file__).resolve().parent.parent
SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def get_skill_config():
    raw = get_config_section("skills")
    raw_config = dict(raw) if isinstance(raw, dict) else {}
    paths = raw_config.get("paths", ["skills"])
    if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
        raise ValueError("skills.paths must be an array of strings")
    config = {"paths": paths}
    allow_call = raw_config.get("allow_call", DEFAULT_ALLOW_CALL)
    if not isinstance(allow_call, bool):
        raise ValueError("skills.allow_call must be a boolean")
    config["allow_call"] = allow_call
    # Resource and tool-loop limits are implementation safety constants, not
    # public configuration. Ignore legacy max_* keys in config files.
    config.update(DEFAULT_LIMITS)
    return config


def _safe_decode(data, label):
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Skill text file is not valid UTF-8: {label}") from exc


def _read_text_resource_candidate(path, relative, max_bytes):
    """Return bounded UTF-8 text bytes, or None when the file is binary."""
    try:
        with path.open("rb") as handle:
            data = handle.read(max_bytes + 1)
    except OSError as exc:
        raise ValueError(f"Unable to read Skill resource: {relative}") from exc

    sample = data[:8192]
    try:
        codecs.getincrementaldecoder("utf-8")().decode(sample, final=False)
    except UnicodeDecodeError:
        return None
    if b"\0" in sample:
        return None
    if len(data) > max_bytes:
        raise ValueError(f"Resource exceeds {max_bytes} bytes: {relative}")
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if b"\0" in data:
        return None
    return data


def _parse_scalar(value):
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def parse_skill_frontmatter(text):
    """Parse the small, string-only YAML subset used by standard skills."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("SKILL.md must start with YAML front matter")
    end = next((index for index in range(1, len(lines)) if lines[index].strip() == "---"), None)
    if end is None:
        raise ValueError("SKILL.md front matter is not terminated")

    result = {}
    index = 1
    while index < end:
        line = lines[index]
        if not line.strip() or line.lstrip().startswith("#"):
            index += 1
            continue
        if line[:1].isspace():
            # Nested values for extension metadata are intentionally ignored;
            # the loader only consumes the standard top-level string fields.
            index += 1
            continue
        if ":" not in line:
            raise ValueError(f"Unsupported SKILL.md front matter at line {index + 1}")
        key, raw_value = line.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if not key:
            raise ValueError(f"Invalid SKILL.md front matter key at line {index + 1}")
        if raw_value in {"|", "|-", "|+", ">", ">-", ">+"}:
            block = []
            index += 1
            while index < end and (not lines[index].strip() or lines[index][:1].isspace()):
                block.append(lines[index].lstrip())
                index += 1
            result[key] = "\n".join(block).strip() if raw_value.startswith("|") else " ".join(
                part.strip() for part in block if part.strip()
            )
            continue
        result[key] = _parse_scalar(raw_value)
        index += 1

    for required in ("name", "description"):
        if not isinstance(result.get(required), str) or not result[required].strip():
            raise ValueError(f"SKILL.md front matter requires non-empty '{required}'")
        result[required] = result[required].strip()
    if len(result["name"]) > 64 or not SKILL_NAME_PATTERN.fullmatch(result["name"]):
        raise ValueError(
            "SKILL.md 'name' must be 1-64 lowercase letters, digits, or single hyphens"
        )
    if len(result["description"]) > 1024:
        raise ValueError("SKILL.md 'description' must not exceed 1024 characters")
    compatibility = result.get("compatibility", "")
    if compatibility is not None and not isinstance(compatibility, str):
        raise ValueError("SKILL.md 'compatibility' must be a string")
    result["compatibility"] = str(compatibility or "").strip()
    if len(result["compatibility"]) > 500:
        raise ValueError("SKILL.md 'compatibility' must not exceed 500 characters")
    return result


def _is_within(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _configured_roots(config):
    roots = []
    for raw in config["paths"]:
        raw = raw.strip()
        if not raw:
            continue
        path = Path(os.path.expandvars(raw)).expanduser()
        if not path.is_absolute():
            path = PLUGIN_ROOT / path
        roots.append(path.resolve())
    return roots


def _find_skill_manifest(directory):
    try:
        matches = [
            item
            for item in directory.iterdir()
            if item.is_file() and item.name.casefold() == "skill.md"
        ]
    except OSError as exc:
        raise ValueError(f"Unable to inspect Skill directory: {directory.name}") from exc
    if len(matches) > 1:
        raise ValueError(f"Skill directory contains multiple SKILL.md files: {directory.name}")
    return matches[0] if matches else None


def _candidate_skill_dirs(config):
    for configured in _configured_roots(config):
        try:
            if not configured.exists() or not configured.is_dir():
                continue
            if _find_skill_manifest(configured) is not None:
                yield configured
                continue
            children = sorted(configured.iterdir(), key=lambda item: item.name.casefold())
        except OSError as exc:
            raise ValueError("Unable to scan a configured Skill root") from exc
        for child in children:
            try:
                is_skill = child.is_dir() and _find_skill_manifest(child) is not None
                resolved = child.resolve() if is_skill else None
            except OSError as exc:
                raise ValueError(f"Unable to inspect Skill directory: {child.name}") from exc
            if is_skill:
                if not _is_within(resolved, configured):
                    raise ValueError(f"Skill directory escapes configured root: {child.name}")
                yield resolved


def _snapshot_skill(skill_root, config):
    root = skill_root.resolve()
    manifest = _find_skill_manifest(root)
    if manifest is None:
        raise ValueError("Skill directory does not contain SKILL.md")
    skill_file = manifest.resolve()
    if not _is_within(skill_file, root):
        raise ValueError("SKILL.md escapes its Skill directory")
    try:
        skill_bytes = skill_file.read_bytes()
    except OSError as exc:
        raise ValueError("Unable to read SKILL.md") from exc
    if len(skill_bytes) > config["max_skill_md_bytes"]:
        raise ValueError(f"SKILL.md exceeds {config['max_skill_md_bytes']} bytes")
    skill_text = _safe_decode(skill_bytes, "SKILL.md")
    metadata = parse_skill_frontmatter(skill_text)

    manifest = []
    total = 0
    try:
        candidates = [
            item for item in root.rglob("*")
            if item.is_file()
            and item != skill_file
            and not any(part.startswith(".") for part in item.relative_to(root).parts)
        ]
    except (OSError, RuntimeError) as exc:
        raise ValueError("Unable to scan Skill resources") from exc

    candidates.sort(key=lambda item: item.relative_to(root).as_posix().casefold())
    for item in candidates:
        relative = item.relative_to(root).as_posix()
        try:
            resolved = item.resolve()
        except (OSError, RuntimeError) as exc:
            raise ValueError(f"Unable to resolve Skill resource: {relative}") from exc
        if not _is_within(resolved, root):
            raise ValueError(f"Resource escapes its Skill directory: {relative}")
        data = _read_text_resource_candidate(
            resolved, relative, config["max_reference_file_bytes"]
        )
        if data is None:
            continue
        total += len(data)
        if total > config["max_reference_total_bytes"]:
            raise ValueError(f"Skill resources exceed {config['max_reference_total_bytes']} bytes")
        manifest.append({
            "path": relative,
            "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        })

    digest_source = {
        "skill_md_sha256": hashlib.sha256(skill_bytes).hexdigest(),
        "references": manifest,
    }
    skill_hash = hashlib.sha256(
        json.dumps(digest_source, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "name": metadata["name"],
        "description": metadata["description"],
        "compatibility": metadata["compatibility"],
        "skill_instructions": skill_text,
        "skill_md_sha256": digest_source["skill_md_sha256"],
        "skill_hash": skill_hash,
        "reference_manifest": manifest,
        "_root": root,
        "_skill_file": skill_file,
    }


def discover_skills(strict=True):
    config = get_skill_config()
    result = []
    names = set()
    errors = []
    for root in _candidate_skill_dirs(config):
        try:
            snapshot = _snapshot_skill(root, config)
            if snapshot["name"] in names:
                raise ValueError(f"Duplicate Skill name: {snapshot['name']}")
            names.add(snapshot["name"])
            result.append(snapshot)
        except Exception as exc:
            if strict or str(exc).startswith("Duplicate Skill name:"):
                raise
            errors.append(str(exc))
    result.sort(key=lambda item: item["name"].casefold())
    return result, errors


def get_skill_summaries():
    """Return only the metadata needed to render the Skill selector."""
    skills, _ = discover_skills(strict=True)
    return [
        {"name": skill["name"], "description": skill["description"]}
        for skill in skills
    ]



def get_skill_snapshot(skill_name):
    skills, _ = discover_skills(strict=True)
    for snapshot in skills:
        if snapshot["name"] == skill_name:
            return snapshot
    raise ValueError(f"Unknown Skill name: {skill_name}")


def create_skill_options(skill_name):
    snapshot = get_skill_snapshot(skill_name)
    return _options_from_snapshot(snapshot)


def _options_from_snapshot(snapshot):
    return {
        "type": SKILL_OPTIONS_TYPE,
        "schema_version": SKILL_OPTIONS_SCHEMA_VERSION,
        "skill_name": snapshot["name"],
        "description": snapshot["description"],
        "compatibility": snapshot["compatibility"],
        "skill_instructions": snapshot["skill_instructions"],
        "skill_hash": snapshot["skill_hash"],
        "reference_manifest": snapshot["reference_manifest"],
    }


def validate_skill_options(options):
    if not isinstance(options, dict):
        raise ValueError("skill_options must come from OpenAI Text Skill Options")
    if options.get("type") != SKILL_OPTIONS_TYPE:
        raise ValueError(f"Unsupported skill_options type: {options.get('type')!r}")
    if options.get("schema_version") != SKILL_OPTIONS_SCHEMA_VERSION:
        raise ValueError(f"Unsupported skill_options schema_version: {options.get('schema_version')!r}")
    name = options.get("skill_name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("skill_options is missing skill_name")
    snapshot = get_skill_snapshot(name)
    expected = _options_from_snapshot(snapshot)
    for key in ("skill_name", "description", "compatibility", "skill_instructions", "skill_hash", "reference_manifest"):
        if options.get(key) != expected[key]:
            raise ValueError(f"Skill changed or skill_options field is invalid: {key}")
    return snapshot



def read_skill_reference(snapshot, relative_path, allow_skill_md=False):
    if not isinstance(relative_path, str):
        raise ValueError("Reference path must be a string")
    if allow_skill_md and relative_path == "SKILL.md":
        skill_file = snapshot["_skill_file"]
        data = skill_file.read_bytes()
        if hashlib.sha256(data).hexdigest() != snapshot["skill_md_sha256"]:
            raise ValueError("Skill reference changed during execution: SKILL.md")
        return {
            "skill_name": snapshot["name"],
            "path": "SKILL.md",
            "sha256": hashlib.sha256(data).hexdigest(),
            "content": _safe_decode(data, "SKILL.md"),
        }
    entry = next((item for item in snapshot["reference_manifest"] if item["path"] == relative_path), None)
    if entry is None:
        return {"error": "reference_not_found", "path": relative_path}
    root = snapshot["_root"].resolve()
    resolved = (root / Path(relative_path)).resolve()
    if not _is_within(resolved, root):
        raise ValueError("Reference path escapes its Skill directory")
    try:
        data = resolved.read_bytes()
    except OSError as exc:
        raise ValueError(f"Unable to read Skill reference: {relative_path}") from exc
    digest = hashlib.sha256(data).hexdigest()
    if len(data) != entry["size"] or digest != entry["sha256"]:
        raise ValueError(f"Skill reference changed during execution: {relative_path}")
    return {
        "skill_name": snapshot["name"],
        "path": relative_path,
        "sha256": digest,
        "content": _safe_decode(data, relative_path),
    }


def load_skill(snapshot):
    """Load and revalidate the complete selected Skill for tool disclosure."""
    result = read_skill_reference(snapshot, "SKILL.md", allow_skill_md=True)
    return {
        "skill_name": snapshot["name"],
        "skill_hash": snapshot["skill_hash"],
        "instructions": result["content"],
        "references": [
            {"path": item["path"], "size": item["size"]}
            for item in snapshot["reference_manifest"]
        ],
    }


LOAD_SKILL_TOOL = "load_skill"
READ_SKILL_FILE_TOOL = "read_skill_file"
INVOCATION_POLICY = "required_once_per_session"
EXECUTION_MODE = "pi_skill_agent"


class SkillExecutionError(ValueError):
    """Stable, traceable failure raised by the Skill execution layer."""

    def __init__(self, code, message, retryable=False):
        self.code = str(code)
        self.message = str(message)
        self.retryable = bool(retryable)
        super().__init__(f"{self.code}: {self.message}")



def _identity(snapshot):
    return {"name": snapshot["name"], "hash": snapshot["skill_hash"]}


def _trace(snapshot=None, protocol=None):
    return {
        "schema_version": 1,
        "execution_mode": EXECUTION_MODE if snapshot else None,
        "protocol": protocol,
        "skill_identity": _identity(snapshot) if snapshot else {},
        "skill_loaded": False,
        "load_source": "none",
        "invocation_policy": INVOCATION_POLICY,
        "tool_choice_mode": None,
        "compat_retry_count": 0,
        "tool_rounds": 0,
        "tool_calls": 0,
        "files_read": [],
        "errors": [],
    }


def _append_trace_error(trace, code, message, **details):
    error = {"code": str(code), "message": str(message)}
    error.update({key: value for key, value in details.items() if value is not None})
    if error not in trace["errors"]:
        trace["errors"].append(error)


def _function_definition(name):
    if name == LOAD_SKILL_TOOL:
        return {
            "name": name,
            "description": "Load the complete selected SKILL.md.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        }
    if name == READ_SKILL_FILE_TOOL:
        return {
            "name": name,
            "description": "Read one exact text resource from anywhere in the loaded Skill directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Exact relative path from the loaded Skill manifest.",
                    }
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        }
    raise ValueError(f"Unknown Skill tool definition: {name}")


def skill_tool_definitions(protocol="openai-completions", loaded=False):
    """Return the provider schema for the only tool registered in this phase."""
    definition = _function_definition(
        READ_SKILL_FILE_TOOL if loaded else LOAD_SKILL_TOOL
    )
    if protocol == "openai-completions":
        return [{"type": "function", "function": definition}]
    if protocol == "openai-responses":
        return [{"type": "function", **definition}]
    raise SkillExecutionError(
        "skill_protocol_not_supported",
        f"Skill execution does not support protocol: {protocol}",
    )


def _parse_args(raw):
    if isinstance(raw, dict):
        value = raw
    else:
        try:
            value = json.loads(raw or "{}")
        except (TypeError, ValueError) as exc:
            raise SkillExecutionError(
                "pi_skill_tool_call_invalid",
                "Skill tool arguments must be valid JSON",
            ) from exc
    if not isinstance(value, dict):
        raise SkillExecutionError(
            "pi_skill_tool_call_invalid",
            "Skill tool arguments must be a JSON object",
        )
    return value


def _sanitize(value, snapshot):
    """Deterministically redact secrets and real local Skill paths."""
    sensitive_keys = {
        "api_key", "authorization", "proxy-authorization", "access_token",
        "secret", "password",
    }
    real_paths = {
        str(snapshot.get("_root", "")),
        str(snapshot.get("_skill_file", "")),
    }
    real_paths.discard("")

    def visit(current):
        if isinstance(current, dict):
            return {
                key: "[REDACTED]" if str(key).casefold() in sensitive_keys else visit(item)
                for key, item in current.items()
            }
        if isinstance(current, list):
            return [visit(item) for item in current]
        if isinstance(current, tuple):
            return [visit(item) for item in current]
        if isinstance(current, str):
            result = current
            for path in sorted(real_paths, key=len, reverse=True):
                result = result.replace(path, "[SKILL_ROOT]")
                result = result.replace(path.replace("\\", "/"), "[SKILL_ROOT]")
            return result
        return current

    return visit(deepcopy(value))


@dataclass
class SkillSession:
    """Append-only ledger used by continuation, replay, and UI output."""

    events: list[dict] = field(default_factory=list)
    version: int = 0
    skill_loaded: bool = False
    skill_identity: dict = field(default_factory=dict)
    load_version: int | None = None
    lock: threading.RLock = field(
        default_factory=threading.RLock, repr=False, compare=False
    )

    def append(self, event_type, **data):
        self.events.append(deepcopy({
            "type": event_type,
            "context_version": self.version,
            **data,
        }))

    def matches_loaded(self, snapshot):
        return self.skill_loaded and self.skill_identity == _identity(snapshot)

    def reset_for(self, snapshot):
        identity = _identity(snapshot)
        if self.skill_identity and self.skill_identity != identity:
            self.skill_loaded = False
            self.load_version = None
            self.append("skill_reset", skill_identity=identity)
        self.skill_identity = identity

    def commit(self, context, snapshot, protocol):
        was_loaded = self.matches_loaded(snapshot)
        self.version += 1
        self.skill_identity = _identity(snapshot)
        self.skill_loaded = True
        if not was_loaded:
            self.load_version = self.version
        self.append(
            "turn_commit",
            protocol=protocol,
            skill_identity=self.skill_identity,
            skill_loaded=True,
            provider_context=context,
        )

    def abort(self, exc):
        self.append(
            "turn_abort",
            error={
                "code": getattr(exc, "code", "pi_skill_execution_aborted"),
                "message": getattr(exc, "message", str(exc)),
            },
        )

    def derive_context(self):
        for event in reversed(self.events):
            if event.get("type") == "skill_reset":
                return []
            if (
                event.get("type") == "turn_commit"
                and event.get("skill_identity") == self.skill_identity
            ):
                return deepcopy(event.get("provider_context") or [])
        return []

    def conversation(self, snapshot):
        status = (
            "committed"
            if self.events and self.events[-1]["type"] == "turn_commit"
            else "aborted"
        )
        return {
            "schema_version": 1,
            "execution_mode": EXECUTION_MODE,
            "context_version": self.version,
            "turn_status": status,
            "skill_identity": dict(self.skill_identity),
            "skill_loaded": self.skill_loaded,
            "load_version": self.load_version,
            "provider_context": _sanitize(self.derive_context(), snapshot),
            "events": _sanitize(self.events, snapshot),
        }


class SkillSessionStore:
    _sessions = {}
    _lock = threading.RLock()

    @classmethod
    def get(cls, key):
        with cls._lock:
            if key not in cls._sessions:
                session = SkillSession()
                session.append("session_open", execution_mode=EXECUTION_MODE)
                cls._sessions[key] = session
            return cls._sessions[key]

    @classmethod
    def clear(cls, key):
        with cls._lock:
            cls._sessions.pop(key, None)


class ReadOnlySkillRuntime:
    """Execute the two registered tools through host file APIs only."""

    def __init__(self, snapshot, limits, trace):
        self.snapshot = snapshot
        self.limits = limits
        self.trace = trace
        self.cache = {}
        self.disclosed = 0
        self.loaded = False

    def _account(self, result):
        added = len(result.get("content", "").encode("utf-8"))
        self.disclosed += added
        if self.disclosed > self.limits["max_disclosed_bytes_per_execution"]:
            raise SkillExecutionError(
                "pi_skill_tool_limit_exceeded", "Skill disclosure byte limit exceeded"
            )
        path = result.get("path")
        if result.get("content") is not None and path not in self.trace["files_read"]:
            self.trace["files_read"].append(path)

    def execute(self, name, arguments):
        if name == LOAD_SKILL_TOOL:
            if arguments:
                raise SkillExecutionError(
                    "pi_skill_tool_call_invalid", "load_skill accepts no arguments"
                )
            if name in self.cache:
                return self.cache[name]
            try:
                output = load_skill(self.snapshot)
            except ValueError as exc:
                raise SkillExecutionError("skill_snapshot_changed", str(exc)) from exc
            self._account({"path": "SKILL.md", "content": output["instructions"]})
            output = _sanitize(output, self.snapshot)
            self.cache[name] = output
            self.loaded = True
            return output

        if name != READ_SKILL_FILE_TOOL:
            raise SkillExecutionError(
                "pi_skill_tool_call_invalid", f"Unknown Skill tool: {name}"
            )
        if not self.loaded:
            raise SkillExecutionError(
                "skill_not_loaded", "read_skill_file requires load_skill first"
            )
        if set(arguments) != {"path"} or not isinstance(arguments.get("path"), str):
            raise SkillExecutionError(
                "pi_skill_tool_call_invalid",
                "read_skill_file requires only the string 'path' argument",
            )
        path = arguments["path"]
        if path == "SKILL.md":
            raise SkillExecutionError(
                "pi_skill_tool_call_invalid", "SKILL.md can only be loaded with load_skill"
            )
        if path in self.cache:
            return self.cache[path]
        try:
            result = read_skill_reference(self.snapshot, path)
        except ValueError as exc:
            raise SkillExecutionError("skill_snapshot_changed", str(exc)) from exc
        if result.get("error"):
            raise SkillExecutionError(
                "pi_skill_tool_call_invalid", f"Resource is not in the manifest: {path}"
            )
        self._account(result)
        self.cache[path] = result
        return result


@dataclass
class NormalizedResponse:
    native_items: list[dict]
    tool_calls: list[dict]
    final_text: str | None


class ProviderAdapter:
    """Wire conversion only. Skill policy remains in PiSkillAgentLoop."""

    protocol = None
    history_field = None

    def __init__(self, post_json, endpoint, headers, timeout, proxies, session):
        self.post_json = post_json
        self.endpoint = endpoint
        self.headers = headers
        self.timeout = timeout
        self.proxies = proxies
        self.session = session

    def _post(self, payload):
        return self.post_json(
            self.endpoint, self.headers, payload, self.timeout, self.proxies
        )

    def request(
        self, base_payload, history, tools, tool_choice=None,
    ):
        raise NotImplementedError

    def normalize(self, data):
        raise NotImplementedError

    def append_native_items(self, history, normalized):
        raise NotImplementedError

    def serialize_tool_result(self, call_id, output):
        raise NotImplementedError


class ResponsesProviderAdapter(ProviderAdapter):
    protocol = "openai-responses"
    history_field = "input"

    def request(
        self, base_payload, history, tools, tool_choice=None,
    ):
        payload = {
            **base_payload,
            "input": deepcopy(history),
            "tools": deepcopy(tools),
            "parallel_tool_calls": False,
        }
        if tool_choice is not None:
            payload["tool_choice"] = deepcopy(tool_choice)
        self.session.append("model_request", protocol=self.protocol, payload=payload)
        try:
            data = self._post(payload)
        except (ToolChoiceRejected, FunctionToolsRejected) as exc:
            raise SkillExecutionError(
                "pi_skill_provider_not_supported",
                "The Responses provider rejected function tool execution",
            ) from exc
        self.session.append("model_response", protocol=self.protocol, response=data)
        return data

    def normalize(self, data):
        if not isinstance(data, dict) or data.get("status") != "completed":
            detail = (
                data.get("incomplete_details") or data.get("error") or data.get("status")
                if isinstance(data, dict)
                else "non-object response"
            )
            raise SkillExecutionError(
                "pi_skill_response_invalid",
                f"Responses request did not complete: {detail or 'unknown'}",
            )
        output = data.get("output")
        if not isinstance(output, list):
            raise SkillExecutionError(
                "pi_skill_response_invalid", "Responses response is missing output"
            )
        calls = []
        chunks = []
        for index, item in enumerate(output):
            if not isinstance(item, dict):
                continue
            if item.get("type") == "function_call":
                calls.append({
                    "type": "tool_call",
                    "id": item.get("id") or item.get("call_id"),
                    "call_id": item.get("call_id"),
                    "name": item.get("name"),
                    "arguments": item.get("arguments"),
                    "native_index": index,
                    "native_item": deepcopy(item),
                })
            elif item.get("type") == "message":
                for block in item.get("content") or []:
                    if (
                        isinstance(block, dict)
                        and block.get("type") == "output_text"
                        and block.get("text")
                    ):
                        chunks.append(block["text"])
        return NormalizedResponse(
            output, calls, "\n".join(chunks) if chunks else None
        )

    def append_native_items(self, history, normalized):
        history.extend(deepcopy(normalized.native_items))

    def serialize_tool_result(self, call_id, output):
        return {
            "type": "function_call_output",
            "call_id": call_id,
            "output": json.dumps(output, ensure_ascii=False),
        }


class CompletionsProviderAdapter(ProviderAdapter):
    protocol = "openai-completions"
    history_field = "messages"

    def request(
        self, base_payload, history, tools, tool_choice=None,
    ):
        payload = {
            **base_payload,
            "messages": deepcopy(history),
            "tools": deepcopy(tools),
            "parallel_tool_calls": False,
        }
        if tool_choice is not None:
            payload["tool_choice"] = deepcopy(tool_choice)
        self.session.append("model_request", protocol=self.protocol, payload=payload)
        try:
            data = self._post(payload)
        except ToolChoiceRejected as exc:
            raise SkillExecutionError(
                "compat_tool_choice_not_supported",
                "The compatible API rejected tool_choice",
            ) from exc
        except FunctionToolsRejected as exc:
            raise SkillExecutionError(
                "compat_function_tools_not_supported",
                "The compatible API rejected Skill function tools",
            ) from exc
        self.session.append("model_response", protocol=self.protocol, response=data)
        return data

    def normalize(self, data):
        choices = data.get("choices") if isinstance(data, dict) else None
        message = (
            choices[0].get("message")
            if isinstance(choices, list)
            and choices
            and isinstance(choices[0], dict)
            else None
        )
        if not isinstance(message, dict):
            raise SkillExecutionError(
                "pi_skill_response_invalid", "Completions response is missing message"
            )
        calls = []
        for index, call in enumerate(message.get("tool_calls") or []):
            function = call.get("function") if isinstance(call, dict) else None
            calls.append({
                "type": "tool_call",
                "id": call.get("id") if isinstance(call, dict) else None,
                "call_id": call.get("id") if isinstance(call, dict) else None,
                "name": function.get("name") if isinstance(function, dict) else None,
                "arguments": function.get("arguments") if isinstance(function, dict) else None,
                "native_index": index,
                "native_item": deepcopy(call),
            })
        content = message.get("content")
        final_text = content if isinstance(content, str) and content.strip() else None
        return NormalizedResponse([deepcopy(message)], calls, final_text)

    def append_native_items(self, history, normalized):
        history.extend(deepcopy(normalized.native_items))

    def serialize_tool_result(self, call_id, output):
        return {
            "role": "tool",
            "tool_call_id": call_id,
            "content": json.dumps(output, ensure_ascii=False),
        }


class SkillToolRegistry:
    """Expose exactly one Skill tool for the current load phase."""

    @staticmethod
    def allowed_name(loaded):
        return READ_SKILL_FILE_TOOL if loaded else LOAD_SKILL_TOOL

    @classmethod
    def validate(cls, call, loaded, seen):
        call_id = call.get("call_id")
        name = call.get("name")
        if not isinstance(call_id, str) or not call_id or call_id in seen:
            raise SkillExecutionError(
                "pi_skill_tool_call_invalid", "Tool call_id is missing or duplicated"
            )
        if name != cls.allowed_name(loaded):
            raise SkillExecutionError(
                "pi_skill_tool_call_invalid",
                f"Tool is not registered in the current phase: {name!r}",
            )
        return call_id, name, _parse_args(call.get("arguments"))


class PiSkillAgentLoop:
    """Protocol-neutral Skill policy, lifecycle, limits, and load gate."""

    def __init__(self, adapter, snapshot, session, limits):
        self.adapter = adapter
        self.snapshot = snapshot
        self.session = session
        self.limits = limits

    def run(self, payload, trace):
        field = self.adapter.history_field
        history = deepcopy(payload[field])
        base_payload = {
            key: deepcopy(value)
            for key, value in payload.items()
            if key not in {field, "tools", "tool_choice", "parallel_tool_calls"}
        }
        loaded = self.session.matches_loaded(self.snapshot)
        runtime = ReadOnlySkillRuntime(self.snapshot, self.limits, trace)
        runtime.loaded = loaded
        if loaded:
            trace["skill_loaded"] = True
            trace["load_source"] = "session"
        trace["tool_choice_mode"] = "provider_default"
        seen = set()

        for round_index in range(self.limits["max_tool_rounds"] + 1):
            tools = skill_tool_definitions(self.adapter.protocol, loaded=loaded)
            data = self.adapter.request(
                base_payload,
                history,
                tools,
            )
            normalized = self.adapter.normalize(data)
            self.adapter.append_native_items(history, normalized)
            self.session.append(
                "model_output_normalized",
                items=[
                    {key: value for key, value in item.items() if key != "native_item"}
                    for item in normalized.tool_calls
                ],
                final_text_present=normalized.final_text is not None,
            )

            if not normalized.tool_calls:
                if not loaded:
                    raise SkillExecutionError(
                        "skill_not_loaded",
                        "The model returned final output before calling load_skill",
                    )
                if normalized.final_text is None:
                    raise SkillExecutionError(
                        "pi_skill_response_invalid",
                        "Provider response contains no final text",
                    )
                trace["skill_loaded"] = True
                if trace["load_source"] == "none":
                    trace["load_source"] = "tool_call"
                self.session.commit(
                    history, self.snapshot, self.adapter.protocol
                )
                return normalized.final_text

            if round_index >= self.limits["max_tool_rounds"]:
                raise SkillExecutionError(
                    "pi_skill_tool_limit_exceeded", "Skill tool round limit exceeded"
                )
            trace["tool_rounds"] += 1
            phase_loaded = loaded
            planned = []
            for call in normalized.tool_calls:
                self.session.append("tool_call_planned", call=call)
                try:
                    call_id, name, arguments = SkillToolRegistry.validate(
                        call, phase_loaded, seen
                    )
                except SkillExecutionError as exc:
                    self.session.append(
                        "tool_call_validated",
                        call_id=call.get("call_id"),
                        name=call.get("name"),
                        accepted=False,
                        error={"code": exc.code, "message": exc.message},
                    )
                    raise
                seen.add(call_id)
                planned.append((call_id, name, arguments))
                self.session.append(
                    "tool_call_validated", call_id=call_id, name=name,
                    arguments=arguments, accepted=True,
                )

            for call_id, name, arguments in planned:
                trace["tool_calls"] += 1
                if trace["tool_calls"] > self.limits["max_tool_calls_per_execution"]:
                    raise SkillExecutionError(
                        "pi_skill_tool_limit_exceeded", "Skill tool call limit exceeded"
                    )
                self.session.append(
                    "tool_call_started", call_id=call_id, name=name,
                    path=arguments.get("path"),
                )
                try:
                    output = runtime.execute(name, arguments)
                except SkillExecutionError as exc:
                    self.session.append(
                        "tool_call_settled", call_id=call_id, name=name,
                        outcome="error", path=arguments.get("path"),
                        error={"code": exc.code, "message": exc.message},
                    )
                    raise
                self.session.append(
                    "tool_call_settled", call_id=call_id, name=name,
                    outcome="success", path=output.get("path"), output=output,
                )
                wire_result = self.adapter.serialize_tool_result(call_id, output)
                history.append(wire_result)
                self.session.append(
                    "tool_result_appended", call_id=call_id,
                    protocol=self.adapter.protocol, item=wire_result,
                )
                if name == LOAD_SKILL_TOOL and runtime.loaded:
                    loaded = True
                    trace["skill_loaded"] = True
                    trace["load_source"] = "tool_call"

        raise SkillExecutionError(
            "pi_skill_tool_limit_exceeded", "Skill tool round limit exceeded"
        )


class SkillExecutionRouter:
    """Activate the Pi Skill Agent only for an explicitly selected Skill."""

    @staticmethod
    def _merge_context(protocol, previous, incoming):
        if not previous:
            return list(incoming)
        current = list(incoming)
        if protocol == "openai-completions":
            while (
                current
                and isinstance(current[0], dict)
                and current[0].get("role") in {"system", "developer"}
            ):
                current.pop(0)
        return list(previous) + current

    @staticmethod
    def _selected_skill_text(snapshot, already_loaded):
        action = (
            "The complete SKILL.md is already present in this Skill session. "
            "Use it for this request and read only needed text resources."
            if already_loaded
            else "Use the selected Skill for this request. Before answering, load the complete SKILL.md."
        )
        return (
            "<selected_skill>\n"
            f"  <name>{html.escape(snapshot['name'])}</name>\n"
            f"  <description>{html.escape(snapshot['description'])}</description>\n"
            "  <location>SKILL.md</location>\n"
            "</selected_skill>\n"
            f"{action}\n"
            "Scripts, network access, shell execution, and file writes are unavailable.\n\n"
        )

    @staticmethod
    def _inject_user_context(protocol, items, snapshot, already_loaded):
        result = deepcopy(items)
        prefix = SkillExecutionRouter._selected_skill_text(snapshot, already_loaded)
        for index in range(len(result) - 1, -1, -1):
            item = result[index]
            if not isinstance(item, dict) or item.get("role") != "user":
                continue
            content = item.get("content")
            if isinstance(content, str):
                item["content"] = prefix + content
            elif isinstance(content, list):
                expected = "text" if protocol == "openai-completions" else "input_text"
                block = next(
                    (
                        entry
                        for entry in content
                        if isinstance(entry, dict) and entry.get("type") == expected
                    ),
                    None,
                )
                if block is None:
                    content.insert(0, {"type": expected, "text": prefix})
                else:
                    block["text"] = prefix + str(block.get("text") or "")
            else:
                raise SkillExecutionError(
                    "pi_skill_response_invalid", "Skill request has no user text content"
                )
            return result
        raise SkillExecutionError(
            "pi_skill_response_invalid", "Skill request has no user message"
        )

    @staticmethod
    def _check_tool_conflicts(payload):
        for tool in payload.get("tools") or []:
            if not isinstance(tool, dict):
                continue
            function = (
                tool.get("function")
                if tool.get("type") == "function" and "function" in tool
                else tool
            )
            name = function.get("name") if isinstance(function, dict) else None
            if name in {LOAD_SKILL_TOOL, READ_SKILL_FILE_TOOL}:
                raise SkillExecutionError(
                    "skill_tool_name_conflict",
                    f"The private Skill tool name is already present: {name}",
                )

    @staticmethod
    def run(
        protocol, post_json, endpoint, headers, timeout, proxies, payload,
        snapshot, session_key, persist_context=True, trace=None,
    ):
        config = get_skill_config()
        if not config.get("allow_call", False):
            raise SkillExecutionError(
                "skill_call_disabled", "Skill calls are disabled by configuration"
            )
        if protocol not in {"openai-completions", "openai-responses"}:
            raise SkillExecutionError(
                "skill_protocol_not_supported",
                f"Skill execution does not support protocol: {protocol}",
            )
        active_trace = trace if isinstance(trace, dict) else _trace(snapshot, protocol)
        active_trace["execution_mode"] = EXECUTION_MODE
        session = SkillSessionStore.get(session_key) if persist_context else SkillSession()
        if not persist_context:
            session.append("session_open", execution_mode=EXECUTION_MODE)

        with session.lock:
            session.reset_for(snapshot)
            loaded_from_session = persist_context and session.matches_loaded(snapshot)
            session.append(
                "skill_manifest", skill_identity=_identity(snapshot),
                metadata={
                    "name": snapshot["name"],
                    "description": snapshot["description"],
                    "location": "SKILL.md",
                },
                references=[
                    {"path": item["path"], "size": item["size"]}
                    for item in snapshot["reference_manifest"]
                ],
            )
            try:
                SkillExecutionRouter._check_tool_conflicts(payload)
                field = "messages" if protocol == "openai-completions" else "input"
                incoming = list(payload[field])
                previous_context = session.derive_context() if persist_context else []
                if previous_context:
                    incoming = SkillExecutionRouter._merge_context(
                        protocol, previous_context, incoming
                    )
                incoming = SkillExecutionRouter._inject_user_context(
                    protocol, incoming, snapshot, loaded_from_session
                )
                session.append(
                    "request_input", protocol=protocol, items=incoming,
                    persist_context=bool(persist_context),
                )
                routed_payload = {**payload, field: incoming}
                adapter_type = (
                    ResponsesProviderAdapter
                    if protocol == "openai-responses"
                    else CompletionsProviderAdapter
                )
                adapter = adapter_type(
                    post_json, endpoint, headers, timeout, proxies, session
                )
                result = PiSkillAgentLoop(
                    adapter, snapshot, session, config
                ).run(routed_payload, active_trace)
                return result, active_trace, session.conversation(snapshot)
            except Exception as exc:
                session.abort(exc)
                code = getattr(exc, "code", "pi_skill_execution_aborted")
                message = getattr(exc, "message", str(exc))
                _append_trace_error(
                    active_trace, code, _sanitize(message, snapshot)
                )
                raise

@dataclass
class SkillRequestContext:
    """Small node-facing facade for the complete Skill request lifecycle."""

    snapshot: dict | None
    protocol: str
    trace: dict

    @classmethod
    def create(cls, options, protocol):
        snapshot = validate_skill_options(options) if options is not None else None
        return cls(snapshot, protocol, _trace(snapshot, protocol if snapshot else None))

    @property
    def enabled(self):
        return self.snapshot is not None

    def session_discriminator(self, system_prompt):
        if not self.enabled:
            return system_prompt
        return json.dumps(
            [system_prompt or "", self.snapshot["skill_hash"], EXECUTION_MODE],
            ensure_ascii=False,
        )

    def validate_advanced_options(self, options):
        if not self.enabled or not isinstance(options, dict):
            return
        forbidden = {
            "tools", "tool_choice", "parallel_tool_calls",
            "previous_response_id", "conversation", "container", "containers",
            "skills",
        }.intersection(options)
        if forbidden:
            raise ValueError(
                "Advanced options cannot override Skill tool fields: "
                + ", ".join(sorted(forbidden))
            )

    def clear_session(self, session_key):
        if self.enabled:
            SkillSessionStore.clear(session_key)

    def execute(
        self, post_json, endpoint, headers, timeout, proxies, payload,
        session_key, persist_context=True,
    ):
        if not self.enabled:
            raise RuntimeError("Skill execution requires selected skill_options")
        try:
            result, _, conversation = SkillExecutionRouter.run(
                self.protocol,
                post_json,
                endpoint,
                headers,
                timeout,
                proxies,
                payload,
                self.snapshot,
                session_key,
                persist_context=persist_context,
                trace=self.trace,
            )
            return result, conversation
        except Exception as exc:
            detail = str(exc)
            if not self.trace["errors"]:
                _append_trace_error(
                    self.trace,
                    getattr(exc, "code", "execution_error"),
                    getattr(exc, "message", detail),
                )
            trace_json = self.trace_json()
            if isinstance(exc, ValueError):
                raise ValueError(f"{detail}\nSkill Trace: {trace_json}") from exc
            raise ValueError(
                f"The API request failed: {detail}\nSkill Trace: {trace_json}"
            ) from exc

    def trace_json(self):
        return json.dumps(self.trace, ensure_ascii=False)
