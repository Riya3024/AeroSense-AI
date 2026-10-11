"""Repository hygiene: secrets, .gitignore, .env.example, AWS templates, Windows script encoding."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "node_modules", "data", "logs"}
TEXT_SUFFIXES = {".py", ".md", ".json", ".ps1", ".ini", ".txt", ".yml", ".yaml", ".toml", ".cfg", ".example"}


def _text_files():
    for p in ROOT.rglob("*"):
        if p.is_file() and not (set(p.relative_to(ROOT).parts) & SKIP_DIRS):
            if p.suffix in TEXT_SUFFIXES or p.name in {".gitignore", ".env.example"}:
                yield p


def _env_example_pairs():
    pairs = {}
    for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Z][A-Z0-9_]*)=(.*)$", line.strip())  # commented-out lines do not match
        if m:
            pairs[m.group(1)] = m.group(2).strip()
    return pairs


def test_no_hardcoded_credentials_anywhere_in_the_repo():
    patterns = {
        "AWS access key id": re.compile(r"\b(AKIA|ASIA)[0-9A-Z]{16}\b"),
        "AWS secret assignment": re.compile(r"aws_secret_access_key\s*[=:]\s*['\"]?[A-Za-z0-9/+=]{30,}", re.I),
        "private key block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    }
    hits = [f"{p.relative_to(ROOT)}: {name}" for p in _text_files()
            for name, rx in patterns.items() if rx.search(p.read_text(encoding="utf-8", errors="ignore"))]
    assert hits == []


def test_gitignore_protects_env_and_credentials_but_keeps_the_template():
    lines = {l.strip() for l in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()}
    for required in (".env", ".env.*", "!.env.example", ".aws/", "*.pem", "*.tfstate"):
        assert required in lines, f".gitignore is missing {required}"


def test_env_example_has_no_secret_values_and_no_aws_access_keys():
    pairs = _env_example_pairs()
    assert "AWS_ACCESS_KEY_ID" not in pairs and "AWS_SECRET_ACCESS_KEY" not in pairs and "AWS_SESSION_TOKEN" not in pairs
    for name, value in pairs.items():
        if re.search(r"KEY|SECRET|TOKEN|PASSWORD", name):
            assert value == "", f"{name} must be blank in .env.example"
    assert pairs["AWS_REGION"] == "ap-south-1"


def test_every_env_var_read_by_backend_settings_is_documented_in_env_example():
    source = (ROOT / "backend" / "settings.py").read_text(encoding="utf-8")
    used = set(re.findall(r'_get\(env, "([A-Z0-9_]+)"', source))
    assert used, "expected to find env var reads in backend/settings.py"
    documented = set(_env_example_pairs()) | set(re.findall(r"^#\s*([A-Z][A-Z0-9_]*)=", (ROOT / ".env.example").read_text(encoding="utf-8"), re.M))
    assert used - documented == {"AWS_DEFAULT_REGION"}  # fallback only; deliberately not advertised


def test_iam_policy_templates_are_valid_json_pinned_to_ap_south_1_with_placeholders():
    files = sorted((ROOT / "aws" / "policies").glob("*.json"))
    assert len(files) == 4
    for f in files:
        doc = json.loads(f.read_text(encoding="utf-8"))
        assert doc["Version"] == "2012-10-17" and doc["Statement"]
        raw = f.read_text(encoding="utf-8")
        for region in re.findall(r"arn:aws:[a-z0-9-]+:([a-z0-9-]*):", raw):
            assert region in ("", "ap-south-1"), f"{f.name} references region {region}"
        assert not re.search(r"arn:aws:[a-z0-9-]*:[a-z0-9-]*:\d{12}:", raw), f"{f.name} contains a real account id"
        for stmt in doc["Statement"]:
            assert stmt["Effect"] == "Allow"


def test_lambda_policy_is_scoped_not_wildcard_admin():
    doc = json.loads((ROOT / "aws" / "policies" / "lambda-execution-policy.json").read_text(encoding="utf-8"))
    for stmt in doc["Statement"]:
        actions = stmt["Action"] if isinstance(stmt["Action"], list) else [stmt["Action"]]
        assert not any(a == "*" or a.endswith(":*") for a in actions)
        assert stmt["Resource"] != "*"


def test_powershell_scripts_are_ascii_only():
    """Windows PowerShell 5.1 reads BOM-less UTF-8 as ANSI, so non-ASCII characters corrupt scripts."""
    scripts = list((ROOT / "scripts").glob("*.ps1")) + list((ROOT / "aws" / "scripts").glob("*.ps1"))
    assert len(scripts) >= 6
    for s in scripts:
        s.read_bytes().decode("ascii")


def test_expected_project_files_exist():
    for rel in ("README.md", "docs/architecture.md", "requirements.txt", "pytest.ini", ".env.example",
                "scripts/run_pipeline.ps1", "data_pipeline/merge_datasets.py"):
        assert (ROOT / rel).exists(), rel
