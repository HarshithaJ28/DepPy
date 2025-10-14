# spdoctor/resolver.py
import requests
import networkx as nx
from packaging.specifiers import SpecifierSet
from packaging.version import Version, InvalidVersion
from typing import List, Tuple, Dict, Optional, Any
from rich.console import Console
from rich.table import Table
from concurrent.futures import ThreadPoolExecutor, as_completed
from . import utils as _utils
import datetime
from . import utils as _utils

console = Console()

PYPI_JSON_URL = "https://pypi.org/pypi/{pkg}/json"

def fetch_pypi_info(package_name: str) -> Optional[Dict[str, Any]]:
    """
    Fetch PyPI JSON metadata for a package (package_name).
    Returns parsed JSON or None on error.
    """
    try:
        resp = requests.get(PYPI_JSON_URL.format(pkg=package_name), timeout=10)
        if resp.status_code == 200:
            return resp.json()
        else:
            return None
    except Exception:
        return None

def parse_requires_dist_field(req_str: str) -> Tuple[str, Optional[str]]:
    """
    Example requires_dist field:
      "numpy (>=1.20.0,<2.0.0)"
      "six"
    Returns (name, specifier_string or None)
    """
    # Common formats: "name (specifier)", "name"
    if "(" in req_str:
        name, rest = req_str.split("(", 1)
        name = name.strip()
        spec = rest.rstrip(")").strip()
        return (name, spec or None)
    else:
        return (req_str.strip(), None)

def build_dependency_graph(top_level_deps: List[Tuple[str, Optional[str]]], max_workers: int = 8, max_depth: Optional[int] = None) -> nx.DiGraph:
    """
    Given top-level dependencies (list of (name, version_spec_or_exact)),
    build a directed graph of dependencies using PyPI metadata.
    Nodes: package names
    Edge attributes:
       - constraint: a string specifier (e.g., ">=1.2,<2.0") or None
       - from: parent package (who requires)
    Note: We fetch requires_dist for the latest release metadata on PyPI.
    """
    G = nx.DiGraph()

    # Seed with top-level nodes
    seed = [ (name, spec) for name, spec in top_level_deps ]
    for name, spec in seed:
        G.add_node(name, top_level=True, pinned=spec)

    # BFS/queue to expand dependencies one layer at a time to avoid runaway recursion
    # queue holds tuples: (pkg_name, depth)
    queue = [(name, 0) for name, _ in seed]
    seen = set([name for name, _ in seed])

    # local in-memory cache to avoid repeated fetches
    local_cache: Dict[str, Optional[Dict[str, Any]]] = {}

    # to make graph building faster, fetch metadata for nodes in parallel in batches
    from concurrent.futures import ThreadPoolExecutor, as_completed

    while queue:
        # collect a small batch from the queue up to max_workers
        batch = []
        while queue and len(batch) < max_workers:
            batch.append(queue.pop(0))

        # prepare nodes for fetch
        to_fetch = []
        for pkg, depth in batch:
            if max_depth is not None and depth > max_depth:
                continue
            if pkg in local_cache:
                continue
            to_fetch.append((pkg, depth))

        # parallel fetch for this batch
        if to_fetch:
            with ThreadPoolExecutor(max_workers=min(max_workers, max(1, len(to_fetch)))) as ex:
                futures = {ex.submit(lambda p: (_utils.get_cached_pypi(p) or fetch_pypi_info(p)), pkg): (pkg, depth) for pkg, depth in to_fetch}
                for fut in as_completed(futures):
                    pkg, depth = futures[fut]
                    try:
                        meta = fut.result()
                        if meta:
                            _utils.cache_pypi_info(pkg, meta)
                        local_cache[pkg] = meta
                    except Exception:
                        local_cache[pkg] = None

        # process this batch (use cached or freshly fetched metadata)
        for pkg, depth in batch:
            if max_depth is not None and depth > max_depth:
                continue
            meta = local_cache.get(pkg)
            if not meta:
                # Keep node but mark as no-meta
                G.nodes[pkg]["pypi_meta_found"] = False
                continue

            G.nodes[pkg]["pypi_meta_found"] = True
            # get requires_dist for latest release info (info.requires_dist)
            requires = meta.get("info", {}).get("requires_dist") or []
            for req in requires:
                dep_name, dep_spec = parse_requires_dist_field(req)
                # add node and edge
                G.add_node(dep_name, top_level=False)
                # edge from pkg -> dep_name with constraint
                G.add_edge(pkg, dep_name, constraint=dep_spec)
                if dep_name not in seen:
                    seen.add(dep_name)
                    queue.append((dep_name, depth + 1))

    return G

def aggregate_constraints_for_node(G: nx.DiGraph, node: str, top_level_pins: Dict[str, Optional[str]]) -> SpecifierSet:
    """
    Collect all constraints that target `node` from incoming edges plus any top-level pin.
    Return a packaging.SpecifierSet representing the intersection.
    """
    specs = []
    # incoming edges: (predecessor, node)
    for pred in G.predecessors(node):
        constraint = G.edges[pred, node].get("constraint")
        if constraint:
            specs.append(constraint)
    # if top-level pinned (like requests==2.31.0) present, include exact version as a specifier
    if node in top_level_pins and top_level_pins[node]:
        pin = top_level_pins[node]
        # if pin looks like exact '2.31.0' or '==2.31.0', normalize
        if pin.startswith("=="):
            specs.append(pin)
        elif any(op in pin for op in [">=", "<=", ">", "<", "~=", "=="]):
            specs.append(pin)
        else:
            # treat plain '2.31.0' as exact
            specs.append(f"=={pin}")

    # build a SpecifierSet from all specs combined
    if not specs:
        return SpecifierSet()
    # intersection is represented by combining spec strings separated by ','
    combined = ",".join(specs)
    try:
        return SpecifierSet(combined)
    except Exception:
        # if packaging can't parse, return a permissive empty set object
        return SpecifierSet()

def find_compatible_versions(pypi_meta: Dict[str, Any], specset: SpecifierSet) -> List[str]:
    """
    Given PyPI metadata dict and a SpecifierSet, return list of version strings that satisfy it.
    We iterate available releases (keys of releases) and test packaging.Version against specset.
    """
    results = []
    if not pypi_meta:
        return results
    releases = list(pypi_meta.get("releases", {}).keys())
    # sort releases by Version (best-effort)
    try:
        releases_sorted = sorted(releases, key=lambda v: Version(v))
    except Exception:
        releases_sorted = releases
    for v in releases_sorted:
        try:
            ver = Version(v)
        except InvalidVersion:
            continue
        if not specset:
            results.append(v)
        else:
            if ver in specset:
                results.append(v)
    return results

def detect_conflicts(G: nx.DiGraph, top_level_deps: List[Tuple[str, Optional[str]]]) -> List[Dict[str, Any]]:
    """
    For each node in graph, aggregate constraints and check available versions on PyPI.
    If no versions satisfy the aggregated SpecifierSet -> record a conflict.
    Return list of conflict dicts with helpful context.
    """
    top_level_pins = {name: pin for name, pin in top_level_deps}
    conflicts = []
    # fetch pypi metadata for all nodes in parallel to reduce latency
    nodes = list(G.nodes())
    meta_map: Dict[str, Optional[Dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=min(8, max(2, len(nodes)))) as ex:
        futures = {ex.submit(lambda n: (_utils.get_cached_pypi(n) or fetch_pypi_info(n)), node): node for node in nodes}
        for fut in as_completed(futures):
            node = futures[fut]
            try:
                meta_map[node] = fut.result()
                if meta_map[node]:
                    _utils.cache_pypi_info(node, meta_map[node])
            except Exception:
                meta_map[node] = None

    for node in nodes:
        pypi_meta = meta_map.get(node)
        specset = aggregate_constraints_for_node(G, node, top_level_pins)
        available = find_compatible_versions(pypi_meta, specset)
        # if there are zero versions satisfying aggregated constraints, it's a conflict
        if pypi_meta is None:
            # could not fetch metadata: skip or optionally warn
            continue
        if not available:
            # construct human-readable context: who requires it and their constraints
            incoming = []
            for pred in G.predecessors(node):
                c = G.edges[pred, node].get("constraint")
                incoming.append({"from": pred, "constraint": c})
            # include top-level pin if present
            pinned = top_level_pins.get(node)
            conflict = {
                "package": node,
                "aggregated_specifier": str(specset) if str(specset) else None,
                "incoming_constraints": incoming,
                "top_level_pin": pinned,
                "available_versions_count": len(pypi_meta.get("releases", {}) if pypi_meta else {}),
            }
            # append to RAG conflict history (best-effort)
            try:
                entry = {"ts": datetime.datetime.utcnow().isoformat() + 'Z', "conflict": conflict}
                _utils.append_conflict_history(entry)
            except Exception:
                pass
            conflicts.append(conflict)
    return conflicts

def pretty_print_report(G: nx.DiGraph, top_level_deps: List[Tuple[str, Optional[str]]], quick: bool = True, show_graph_table: bool = True):
    """
    Print a dependency graph summary.
    - quick=True: produce a fast, attractive summary without fetching PyPI metadata (no network calls).
      Returns None.
    - quick=False: run the full detection (may fetch PyPI metadata) and return conflicts list.
    """
    from rich.panel import Panel
    from rich.columns import Columns
    from rich.syntax import Syntax

    node_count = len(G.nodes())
    edge_count = len(G.edges())

    header = f"[bold magenta]Dependency Graph[/bold magenta] — Nodes: [bold]{node_count}[/bold], Edges: [bold]{edge_count}[/bold]"

    # Quick summary (no network): show top-level deps and high-degree nodes
    if quick:
        console.print(Panel(header, expand=False))

        # Top-level deps as compact bullets
        top_lines = []
        for name, pin in top_level_deps:
            pin_str = pin or "latest"
            top_lines.append(f"[green]•[/green] {name} [dim]{pin_str}[/dim]")
        console.print(Panel(Columns(top_lines, equal=True, expand=True), title="Top-level dependencies", expand=False))

        # show top 8 nodes by degree to surface hotspots
        deg = sorted(G.degree(), key=lambda x: x[1], reverse=True)
        hotspot_lines = []
        for n, d in deg[:8]:
            meta_flag = G.nodes[n].get("pypi_meta_found")
            badge = "✅" if meta_flag else "⚠️"
            hotspot_lines.append(f"{badge} {n} ([bold]{d}[/bold])")
        if hotspot_lines:
            console.print(Panel(Columns(hotspot_lines, equal=True), title="Hotspots (degree)", expand=False))

        console.print("[dim]Quick summary complete — run full detection for detailed conflict output.[/dim]")
        return None

    # Full report (may perform network calls)
    if show_graph_table:
        console.print("[bold magenta]Dependency Graph Summary[/bold magenta]")
        console.print(f"Nodes: {node_count}  Edges: {edge_count}")
        # show top-level deps table
        t = Table(show_header=True, header_style="bold")
        t.add_column("Top-level package")
        t.add_column("Pinned")
        for name, pin in top_level_deps:
            t.add_row(name, pin or "latest")
        console.print(t)

    # Detect conflicts using resolver logic and return them to the caller for unified reporting.
    if show_graph_table:
        console.print("\n[bold magenta]Detecting conflicts via PyPI metadata[/bold magenta]")
    conflicts = detect_conflicts(G, top_level_deps)
    # Return conflicts list; caller (CLI) will handle printing a unified report.
    return conflicts
