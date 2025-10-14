# DepPy — Smart Package Doctor

DepPy (Smart Package Doctor) is a deterministic dependency conflict detector and resolver for Python projects. It analyzes a requirements-style file, builds a dependency graph using PyPI metadata, detects version conflicts between top-level pins and transitive requirements, and produces a pinned `requirements_final.txt` with suggested fixes. The tool emphasizes determinism (heuristics + MILP/OR-Tools solvers), reproducible suggestions, and fast single-command operation.

---

## Highlights

- Deterministic hybrid resolver (heuristic + MILP + OR-Tools CP-SAT) for robust package selection.
- Fast graph building with optional concurrency and in-memory caching to speed up runs.
- Clear, human-friendly CLI output (grouped aggregated conflicts, suggestions, final changes).
- Designed to be used as a one-liner in CI or locally to sanitize and pin a requirements file.
- Extensible architecture: separate modules for parsing, graph resolution, candidate collection, solver orchestration, and utilities.

---

## Key Features

- Parse `requirements.txt`-style files and identify top-level dependencies.
- Build a transitive dependency graph using PyPI metadata.
- Detect conflicts where top-level pins violate the specifiers required by dependents.
- Suggest deterministic upgrades/downgrades or pins using solver-backed optimization.
- Generate `requirements_final.txt` with pinned versions.
- CLI flags: `--workers` (concurrency tuning), `--max-depth` (limit graph expansion), `suggest` subcommand with `--auto-fix` and `--out` options.

---

## Tech Stack

- Python 3.10+ (developed and tested in a virtual environment)
- NetworkX: dependency graph modeling
- packaging: version and specifier parsing
- OR-Tools (optional): CP-SAT solver for constrained selection
- PuLP (optional): MILP fallback solver

## Tech stack & why it matters

- Python 3.10+ — modern typing and ecosystem compatibility.
- NetworkX — models dependency graphs cleanly for traversal and constraint propagation.
- packaging (pip) — robust, standards-compliant parsing of versions and specifiers.
- OR-Tools (CP-SAT) — high-performance constraint solver for selection problems (optional; strong for complex graphs).
- PuLP (CBC MILP) — robust fallback optimization for environments without OR-Tools.
- requests — simple, reliable PyPI metadata fetches.
- rich — professional CLI output that helps humans review decisions quickly.
- pytest — automated tests and regression checks.

These choices show real-world engineering skills: graph algorithms, optimization modeling, API integration, concurrency, UX for terminal tools, and testing.

---

## Quickstart

Two simple ways to use the CLI depending on if you want a minimal run or advanced control.

Minimal (one-liner)

```powershell
# Minimal: auto-detects concurrency and runs a full analysis
python -m spdoctor.cli requirements.txt
```

This repository includes a bundled virtual environment named `spvenv`, you can activate it and run the CLI immediately without installing dependencies:

PowerShell (Windows):

```powershell
# activate bundled venv (Windows PowerShell)
.\spvenv\Scripts\Activate.ps1
python -m spdoctor.cli requirements.txt
```

macOS / Linux (bash/zsh):

```bash
# activate bundled venv (Unix)
source spvenv/bin/activate
python -m spdoctor.cli requirements.txt
```

Advanced (explicit flags)

```powershell
# Advanced: control workers, traversal depth, and output
python -m spdoctor.cli requirements.txt --workers 16 --max-depth 3

# Use the suggest subcommand and write an output file
python -m spdoctor.cli suggest requirements.txt --auto-fix --out fixed_requirements.txt
```

What these options do:
- `--workers N` — set the number of concurrent network workers (overrides auto-detect)
- `--max-depth N` — limit how deep the tool will expand transitive dependencies
- `--auto-fix` — (used with `suggest`) automatically write the suggested pinned file
- `--out <path>` — (used with `suggest`) specify the output path for the fixed requirements file

---

## Example output (what you'll see)

- Grouped "Aggregated conflicts" panel with per-package entries and indented incoming constraints. Easy to scan.
- A "Suggestions" panel listing recommended actions (upgrade/pin/keep) with reasoning.
- A final "Suggested changes (final)" panel with the final pinned map.

This structure lets a reviewer quickly see the problem, the proposed solution, and the final result to apply.

---

## Integration: use in CI or other tools

Add this as a single step in GitHub Actions, GitLab CI, or any pipeline to verify/pin your deps before deployment.

GitHub Actions minimal example:

```yaml
name: pin-dependencies
on: [push]
jobs:
  pin:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install and run Smart Package Doctor
        run: |
          python -m pip install -r requirements.txt
          python -m spdoctor.cli requirements.txt
```

Programmatic use (import core functions):

```python
from spdoctor.parser import parse_dependencies
from spdoctor.resolver import build_dependency_graph
from spdoctor.ai_suggester import final_resolve

deps = parse_dependencies('requirements.txt')
G = build_dependency_graph(deps, max_workers=8)
result = final_resolve(aggregated_conflicts, deps)
```

---

## Developer setup

PowerShell quick setup (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Note: if the project already contains a bundled `spvenv` folder, you can skip `pip install -r requirements.txt` and simply activate `spvenv` as shown in the Quickstart section.

Run tests:

```powershell
pytest -q
```

Style and lint (suggested):

```powershell
pip install black flake8
black .
flake8 spdoctor
```

---

## Architecture & design (brief, technical)

- Parser — robustly parses typical `requirements.txt` lines and normalizes package names.
- Graph builder (`resolver.py`) — BFS/DFS expansion of transitive dependencies using PyPI metadata (batched parallel fetches and in-memory cache).
- Conflict detection — combines analyzer-based checks and resolver-aggregated views.
- Candidate collection (`ai_suggester.py`) — collects top candidate versions per package, uses RAG-style weighting from historical conflict data.
- Solver orchestration — runs MILP and CP-SAT in parallel, scores feasible solutions, and picks the best according to deterministic rules.

This demonstrates system design, algorithms, and practical trade-offs between speed, determinism, and human interpretability.

---