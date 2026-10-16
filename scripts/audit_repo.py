"""Pre-share audit for this repository. Run from the project root:  python scripts/audit_repo.py"""
import json, re, subprocess, sys


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(("PASS  " if ok else "FAIL  ") + name + (f"   [{detail}]" if detail else ""))


if not sh("git rev-parse --show-toplevel"):
    sys.exit("Run this from inside the repository.")

subprocess.run("git fetch -q", shell=True)
status = sh("git status -sb").splitlines()[0]
check("Local copy and GitHub are in sync", "ahead" not in status and "behind" not in status, status)
check("No uncommitted changes", sh("git status --porcelain") == "")

remote = sh("git ls-tree -r --name-only origin/main").splitlines()
for f in ["README.md", "LICENSE", "requirements.txt", ".gitignore", "docs/report.png"]:
    check(f"{f} is on GitHub", f in remote)

junk = [f for f in remote if f.startswith(".venv/") or "__pycache__" in f or f.endswith(".jsonl") or f == "report.html"]
check("No junk committed (.venv, caches, logs, report.html)", not junk, ", ".join(junk[:3]))

rows = [l.split() for l in sh("git ls-tree -r -l origin/main").splitlines()]
big = max(((int(r[3]), r[4]) for r in rows if r[3].isdigit()), default=(0, ""))
check("No file over 2 MB", big[0] < 2_000_000, f"largest: {big[1]} {big[0] // 1024} KB")

readme = sh("git show origin/main:README.md")
imgs = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", readme)
check("Every README image exists on GitHub", bool(imgs) and all(i in remote for i in imgs), f"{len(imgs)} image(s)")
check("README has the real-data results section", "Example results on real data" in readme)

out = sh(f"{sys.executable} -m pytest -q 2>&1 | tail -1")
m = re.search(r"(\d+) passed", out)
check("All tests pass", bool(m) and "failed" not in out and "error" not in out.lower(), out)
claimed = re.search(r"(\d+) tests", readme)
if m and claimed:
    check("README test count matches reality", m.group(1) == claimed.group(1), f"README says {claimed.group(1)}, actual {m.group(1)}")

sys.path.insert(0, ".")
try:
    from quantkit.securebot import scan_for_secrets
    hits = scan_for_secrets(".")
    check("No hardcoded secrets found", not hits, str(hits[:2]))
except Exception as e:
    check("Secret scan ran", False, str(e))

authors = sh("git log --format='%an <%ae>' | sort -u").splitlines()
bad = [a for a in authors if "example.com" in a]
check("Commit authors look right (no placeholder email)", bool(authors) and not bad, "; ".join(authors))

try:
    meta = json.loads(sh("gh repo view --json description,repositoryTopics,visibility,licenseInfo"))
    topics = [t.get("name") or t.get("topic", {}).get("name") for t in (meta.get("repositoryTopics") or [])]
    check("Repo is public", str(meta.get("visibility", "")).upper() == "PUBLIC", meta.get("visibility"))
    check("Description is set", bool(meta.get("description")))
    check("Topics are set", len(topics) >= 3, ", ".join(t for t in topics if t))
    check("GitHub recognises the license", (meta.get("licenseInfo") or {}).get("key") == "mit")
except Exception as e:
    check("Read repo details with gh", False, str(e))

print()
print(f"{sum(results)} of {len(results)} checks passed")
print("Still check by eye: run  open docs/report.png  (only the report, no browser tab bar).")
sys.exit(0 if all(results) else 1)
