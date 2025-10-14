# spdoctor/cli.py
from spdoctor.parser import parse_dependencies
from spdoctor.ai_suggester import suggest_fixes, final_resolve
from spdoctor.conflict_analyzer import check_conflicts, report_conflicts
from spdoctor.resolver import build_dependency_graph, pretty_print_report
from rich.console import Console
import sys
import os

console = Console()


def auto_detect_workers() -> int:
    """Return a sensible default worker count based on CPU cores.
    Keeps values in a reasonable range to avoid overwhelming the machine or PyPI.
    """
    try:
        cpu = os.cpu_count() or 1
    except Exception:
        cpu = 1
    # heuristics: 4x cores, between 2 and 32
    return min(32, max(2, cpu * 4))

def main():
    # Support a simple `suggest` subcommand: `python -m spdoctor.cli suggest <file> [--auto-fix] [--out file]`
    if len(sys.argv) >= 2 and sys.argv[1] == "suggest":
        import argparse
        parser = argparse.ArgumentParser(prog="spdoctor suggest", description="Suggest fixes for dependency conflicts")
        parser.add_argument("file", help="Dependency file to analyze")
        parser.add_argument("--auto-fix", action="store_true", help="Write a fixed requirements file automatically")
        parser.add_argument("--out", default="requirements_fixed.txt", help="Output path for fixed requirements")
        parser.add_argument("--solver", choices=["ortools", "milp", "none"], default="ortools", help="Resolver to use for final selection (ortools preferred)")
        parser.add_argument("--workers", type=int, default=None, help="Number of worker threads to use for network fetches (auto-detected if omitted)")
        parser.add_argument("--max-depth", type=int, default=None, help="Maximum dependency graph depth to expand (optional)")
        args = parser.parse_args(sys.argv[2:])
        if args.workers is None:
            args.workers = auto_detect_workers()
        suggest_command(args.file, auto_fix=args.auto_fix, out_path=args.out, solver=args.solver, workers=args.workers, max_depth=args.max_depth)
        return

    # Single-argument shorthand: run the full suggest flow (scan + resolve + write file)
    if len(sys.argv) >= 2:
        # single-argument shorthand: allow optional flags after file path
        import argparse
        parser = argparse.ArgumentParser(prog="spdoctor", add_help=False)
        parser.add_argument("file", help="Dependency file to analyze")
        parser.add_argument("--workers", type=int, default=None)
        parser.add_argument("--max-depth", type=int, default=None)
        # parse only known args; ignore extras
        args, _ = parser.parse_known_args(sys.argv[1:])
        # If workers wasn't supplied, auto-detect a sensible default
        if args.workers is None:
            args.workers = auto_detect_workers()
        file_path = args.file

        if not os.path.exists(file_path):
            console.print(f"[bold red]Error: File not found: {file_path}[/bold red]")
            sys.exit(1)

        console.print(f"🔍 Scanning and resolving: {file_path}")
        # Read deps and immediately run final resolution with defaults
        deps = parse_dependencies(file_path)
        if not deps:
            console.print("[bold red]No dependencies found or failed to parse.[/bold red]")
            sys.exit(1)

        console.print(f"[bold green]✅ Found {len(deps)} dependencies — summary below:[/bold green]")
        # Quick summary (no network calls) — measure build time
        import time
        t0 = time.perf_counter()
        G = build_dependency_graph(deps, max_workers=args.workers, max_depth=args.max_depth)
        t1 = time.perf_counter()
        pretty_print_report(G, deps, quick=True)
        console.print(f"[dim]Graph built in {t1-t0:.2f}s (workers={args.workers}, max_depth={args.max_depth})[/dim]")
        # proceed to full resolution and write output — reuse the prebuilt graph to avoid duplicate work
        console.print("[bold green]Running full resolution (this may fetch PyPI metadata)...[/bold green]")
        suggest_command(file_path, auto_fix=True, out_path="requirements_final.txt", solver="ortools", prebuilt_graph=G, workers=args.workers, max_depth=args.max_depth)
        return

    # If we reach here, bad usage
    console.print("[bold red]Usage: python -m spdoctor.cli <dependency_file> or python -m spdoctor.cli suggest <dependency_file> [--auto-fix] [--out file] [--solver ortools|milp|none][/bold red]")
    sys.exit(1)


def _aggregate_conflicts(resolver_conflicts, analyzer_conflicts):
    """Aggregate and normalize conflicts from resolver and analyzer into resolver-style shape."""
    from packaging.specifiers import SpecifierSet

    groups = {}

    def add_incoming(pkg, parent, constraint, found):
        key = (pkg.lower(), str(found) if found is not None else None)
        entry = groups.setdefault(key, {"package": pkg, "top_level_pin": found, "incoming_constraints": {}})
        # normalize constraint string
        norm = None
        if constraint:
            try:
                ss = SpecifierSet(constraint)
                norm = str(ss)
            except Exception:
                norm = "".join(constraint.split())
        else:
            norm = ""
        parent_map = entry["incoming_constraints"]
        parent_map.setdefault(parent, set()).add(norm)

    for rc in (resolver_conflicts or []):
        pkg = rc.get("package")
        top_pin = rc.get("top_level_pin")
        incoming = rc.get("incoming_constraints") or []
        if not incoming:
            # Use None for unknown parent so downstream formatting can choose a friendly label
            add_incoming(pkg, None, rc.get("aggregated_specifier") or "", top_pin)
        else:
            for inc in incoming:
                parent = inc.get("from")
                constraint = inc.get("constraint") or rc.get("aggregated_specifier") or ""
                add_incoming(pkg, parent, constraint, top_pin)

    for ac in (analyzer_conflicts or []):
        parent = ac.get("parent")
        requires = ac.get("requires") or ""
        found = ac.get("found")
        parts = requires.split(None, 1)
        if parts:
            target = parts[0]
            constraint = parts[1] if len(parts) > 1 else ""
        else:
            continue
        add_incoming(target, parent, constraint, found)

    aggregated = []
    for (pkg_lower, found), entry in groups.items():
        incoming_map = entry["incoming_constraints"]
        incoming_list = []
        for parent, constraints in incoming_map.items():
            cons = ", ".join(sorted([c for c in constraints if c])) or None
            incoming_list.append({"from": parent, "constraint": cons})
        aggregated.append({"package": entry["package"], "incoming_constraints": incoming_list, "top_level_pin": entry["top_level_pin"]})

    return aggregated

def suggest_command(file_path: str, auto_fix: bool = False, out_path: str = "requirements_fixed.txt", solver: str = "ortools", prebuilt_graph=None, workers: int = 8, max_depth: int = None):
    deps = parse_dependencies(file_path)
    if not deps:
        console.print("[bold red]No dependencies found or failed to parse.[/bold red]")
        return

    # build graph and detect conflicts as in scan
    # reuse prebuilt graph if provided to avoid duplicate work
    G = prebuilt_graph if prebuilt_graph is not None else build_dependency_graph(deps, max_workers=workers, max_depth=max_depth)
    # full detection: ask pretty_print_report to perform conflict detection (quick=False)
    # but when reusing a prebuilt graph we don't want the full dependency table printed again
    resolver_conflicts = pretty_print_report(G, deps, quick=False, show_graph_table=False) or []
    analyzer_conflicts = check_conflicts(deps) or []
    aggregated = _aggregate_conflicts(resolver_conflicts, analyzer_conflicts)

    if not aggregated:
        console.print("[bold green]✅ No conflicts detected.[/bold green]")
        return
    # Use deterministic resolver path (heuristics + RAG cache + solver)
    result = final_resolve(aggregated, deps, auto_fix=auto_fix, out_path=out_path, solver=solver, workers=workers)
    suggestions = result.get("base", {}).get("suggestions", [])
    changes = result.get("final_changes", {})
    gen = result.get("final_file")

    # Pretty print — stacked panels (one below the other) for readability
    from rich.panel import Panel
    from rich.table import Table
    from rich.align import Align

    console.print(Align.center("[bold red]🚨 Conflicts detected![/bold red] Generating suggestions..."))

    # Aggregated conflicts — grouped, human-readable layout
    # For each package show a header: name [top_pin]
    # then list each incoming constraint on its own indented line: - parent: constraint
    grouped_lines = []
    for c in aggregated:
        pkg = c.get('package')
        pin = c.get('top_level_pin') or 'none'
        grouped_lines.append(f"{pkg} [{pin}]")
        incoming = c.get('incoming_constraints', [])
        if not incoming:
            grouped_lines.append("  - (no incoming constraints)")
        else:
            for inc in incoming:
                parent = inc.get('from') or '(unknown)'
                constraint = inc.get('constraint') or '*'
                grouped_lines.append(f"  - {parent}: {constraint}")

    if not grouped_lines:
        grouped_lines = ["No aggregated conflicts detected."]
    console.print(Panel('\n'.join(grouped_lines), title="Aggregated conflicts", border_style="red", expand=True))

    # Suggestions — compact list
    suggestion_lines = []
    for s in suggestions:
        parent = s.get('parent') or '(heuristic)'
        target = s.get('target') or s.get('package') or ''
        action = s.get('suggestion') or s.get('version') or ''
        suggestion_lines.append(f"• {parent} -> {target}: {action}")
    if not suggestion_lines:
        suggestion_lines = ["No suggestions available."]
    console.print(Panel('\n'.join(suggestion_lines), title="Suggestions", border_style="green", expand=True))

    # Final changes — simple list
    change_lines = []
    for pkg, ver in sorted(changes.items()):
        change_lines.append(f"• {pkg} -> {ver}")
    if not change_lines:
        change_lines = ["No final changes."]
    console.print(Panel('\n'.join(change_lines), title="Suggested changes (final)", border_style="cyan", expand=True))

    # Footer: generated file notice
    if gen:
        console.print(Align.center(f"[bold green]✅ Generated:[/bold green] {gen}"))

if __name__ == "__main__":
    main()