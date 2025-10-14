import json
from typing import List, Tuple, Dict, Any, Optional
from packaging.version import Version, InvalidVersion
from packaging.specifiers import SpecifierSet
from rich.console import Console

from . import resolver

console = Console()


def _find_best_version_for_spec(pkg_name: str, spec: Optional[str]) -> Optional[str]:
    """Return the highest PyPI release that satisfies spec (if spec provided), else latest."""
    meta = resolver.fetch_pypi_info(pkg_name)
    if not meta:
        return None
    releases = list(meta.get("releases", {}).keys())
    versions = []
    for r in releases:
        try:
            versions.append(Version(r))
        except InvalidVersion:
            continue
    if not versions:
        return None
    versions_sorted = sorted(versions)
    if not spec:
        return str(versions_sorted[-1])
    try:
        specset = SpecifierSet(spec)
        candidates = [v for v in versions_sorted if v in specset]
        if candidates:
            return str(candidates[-1])
        # if none satisfy, return latest anyway
        return str(versions_sorted[-1])
    except Exception:
        # fallback: try to parse common forms like '>=1.2.3'
        try:
            specset = SpecifierSet(spec.replace(' ', ''))
            candidates = [v for v in versions_sorted if v in specset]
            if candidates:
                return str(candidates[-1])
        except Exception:
            pass
    return str(versions_sorted[-1])


def suggest_fixes(conflicts: List[Dict[str, Any]], top_level_deps: List[Tuple[str, Optional[str]]], auto_fix: bool = False, out_path: str = "requirements_fixed.txt") -> Dict[str, Any]:
    """
    Suggest fixes for conflicts.
    conflicts: list of normalized conflicts in either analyzer or resolver-aggregated shape
    top_level_deps: list of (name, pin)
    Returns dict: {suggestions: [...], changes: {pkg: new_version}, generated_file: path or None}
    """
    # Build top-level map
    top_map = {name.lower(): pin for name, pin in top_level_deps}

    suggestions = []
    changes: Dict[str, str] = {}

    # Normalize incoming shapes to (parent, requires, found) where requires has target and spec
    norm_conflicts = []
    for c in conflicts:
        if 'parent' in c and 'requires' in c and 'found' in c:
            norm_conflicts.append(c)
        elif 'package' in c:
            pkg = c.get('package')
            pin = c.get('top_level_pin')
            for inc in c.get('incoming_constraints', []) or []:
                norm_conflicts.append({
                    'parent': inc.get('from'),
                    'requires': f"{pkg} {inc.get('constraint') or ''}".strip(),
                    'found': pin,
                })

    for c in norm_conflicts:
        parent = c.get('parent')
        requires = c.get('requires') or ""
        found = c.get('found')
        # parse requires to extract target and spec
        parts = requires.split(None, 1)
        target = parts[0] if parts else None
        spec = parts[1] if len(parts) > 1 else None
        if not target:
            continue

        # recommend a version
        best = _find_best_version_for_spec(target, spec)
        if best is None:
            suggestions.append({"target": target, "suggestion": None, "reason": "No PyPI metadata"})
            continue

        # If top-level exists and differs, suggest change, else suggest pin
        top_pin = top_map.get(target.lower())
        action = None
        if top_pin:
            try:
                if top_pin.startswith('=='):
                    current_ver = Version(top_pin[2:])
                else:
                    current_ver = Version(top_pin)
                best_ver = Version(best)
                if best_ver > current_ver:
                    action = f"Upgrade {target} -> {best}"
                    changes[target] = best
                elif best_ver < current_ver:
                    # suggest downgrades rarely; prefer keeping
                    action = f"Keep {target} at {top_pin} (no upgrade suggested)"
                else:
                    action = f"Keep {target} at {top_pin}"
            except Exception:
                action = f"Pin {target} -> {best}"
                changes[target] = best
        else:
            action = f"Pin {target} -> {best}"
            changes[target] = best

        suggestions.append({"parent": parent, "target": target, "requires": spec, "found": found, "suggestion": action, "version": best})

    generated_file = None
    if auto_fix and changes:
        # Create requirements_fixed.txt with minimal changes: pin top-level packages to suggested versions
        lines = []
        for name, pin in top_level_deps:
            lower = name.lower()
            if name in changes or lower in {k.lower(): v for k, v in changes.items()}:
                new_version = changes.get(name) or changes.get(name.capitalize()) or changes.get(lower)
                if new_version:
                    lines.append(f"{name}=={new_version}\n")
                    continue
            if pin:
                lines.append(f"{name}=={pin if pin.startswith('==') else pin}\n")
            else:
                lines.append(f"{name}\n")

        try:
            with open(out_path, 'w') as f:
                f.writelines(lines)
            generated_file = out_path
        except Exception as e:
            console.print(f"[red]Failed to write fixed requirements: {e}[/red]")

    return {"suggestions": suggestions, "changes": changes, "generated_file": generated_file}


### LLM integration and final resolution
### RAG cache helpers
import os
from . import utils as _utils


def _collect_candidates(packages: List[str], top_level_map: Dict[str, Optional[str]], max_per_package: int = 5, workers: int = 8) -> Dict[str, List[str]]:
    """For each package name, fetch PyPI releases and return up to max_per_package candidate versions (sorted ascending).
    Uses RAG cache if available to avoid repeated PyPI requests.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    candidates: Dict[str, List[str]] = {}

    # preload cached metadata for all packages to reduce disk I/O and network calls
    cached_meta: Dict[str, Any] = {}
    for pkg in packages:
        cm = _utils.get_cached_pypi(pkg)
        if cm:
            cached_meta[pkg] = cm

    def _fetch_meta(pkg: str):
        # prefer in-memory cached metadata, then disk cache, then network
        meta = cached_meta.get(pkg)
        if not meta:
            meta = _utils.get_cached_pypi(pkg) or resolver.fetch_pypi_info(pkg)
            if meta:
                # store to both memory and disk cache
                cached_meta[pkg] = meta
                _utils.cache_pypi_info(pkg, meta)
        return pkg, meta

    # parallel fetch
    max_workers = min(workers, max(2, len(packages)))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_fetch_meta, pkg): pkg for pkg in packages}
        for fut in as_completed(futures):
            pkg, meta = fut.result()
            if not meta:
                candidates[pkg] = []
                continue
            releases = list(meta.get("releases", {}).keys())
            vers = []
            for r in releases:
                try:
                    vers.append(Version(r))
                except Exception:
                    continue
            if not vers:
                candidates[pkg] = []
                continue
            vers_sorted = sorted(vers)
            # pick last max_per_package versions
            picked = [str(v) for v in vers_sorted[-max_per_package:]]
            # ensure top-level pin included
            pin = top_level_map.get(pkg.lower())
            if pin:
                pin_ver = pin[2:] if pin.startswith('==') else pin
                if pin_ver not in picked:
                    picked.insert(0, pin_ver)
            candidates[pkg] = picked

    return candidates


### OR-Tools CP-SAT solver integration (preferred)
def solve_ortools(candidates: Dict[str, List[str]], deps_graph: Any, top_level_pins: Dict[str, Optional[str]], rag_weights: Optional[Dict[str, Dict[str, float]]] = None, milp_solution: Optional[Dict[str, str]] = None, time_limit: float = 10.0) -> Optional[Dict[str, str]]:
    """
    Solve selection using OR-Tools CP-SAT. Returns dict pkg->version or None if infeasible or ortools not installed.
    Accepts optional RAG weights and a MILP suggestion map to bias the objective.
    """
    try:
        from ortools.sat.python import cp_model # type: ignore
    except Exception:
        return None

    model = cp_model.CpModel()
    vars_map = {}

    # create boolean var for each candidate
    for pkg, vers in candidates.items():
        for v in vers:
            var = model.NewBoolVar(f"x_{pkg}_{v.replace('.', '_').replace('-', '_')}")
            vars_map[(pkg, v)] = var

    # Exactly one version per package (if candidates exist)
    for pkg, vers in candidates.items():
        if not vers:
            continue
        model.Add(sum(vars_map[(pkg, v)] for v in vers) == 1)

    # Forbidden pairs for incompatible versions
    for parent in deps_graph.nodes():
        for dep in deps_graph.successors(parent):
            constraint = deps_graph.edges[parent, dep].get('constraint')
            if not constraint:
                continue
            dep_cands = candidates.get(dep, [])
            parent_cands = candidates.get(parent, [])
            if not dep_cands or not parent_cands:
                continue
            try:
                specset = SpecifierSet(constraint)
            except Exception:
                specset = None
            for pv in parent_cands:
                for dv in dep_cands:
                    violates = False
                    if specset:
                        try:
                            if Version(dv) not in specset:
                                violates = True
                        except Exception:
                            violates = False
                    if violates:
                        model.Add(vars_map[(parent, pv)] + vars_map[(dep, dv)] <= 1)

    # Objective: combine change penalty, RAG weights, and small MILP preference
    numeric_coeffs = []
    var_list = []
    for pkg, vers in candidates.items():
        # support case-insensitive top-level pins
        pin = top_level_pins.get(pkg) if pkg in top_level_pins else top_level_pins.get(pkg.lower())
        for v in vers:
            # base weight: 1 if changing from top-level pin else 0 (scaled)
            weight = 0
            if pin:
                pin_ver = pin[2:] if pin.startswith('==') else pin
                if v != pin_ver:
                    weight = 100  # scaled weight to prioritize keeping pins
            else:
                weight = 100

            # RAG adjustment: reduce penalty if RAG indicates historical stability
            rag_adj = 1.0
            if rag_weights and isinstance(rag_weights, dict):
                rag_adj = rag_weights.get(pkg, {}).get(v, 1.0)
            # milp bonus: if milp_solution suggests this version, slightly prefer it
            milp_bonus = 0
            if milp_solution and milp_solution.get(pkg) == v:
                milp_bonus = -5

            # final coefficient (integer since CP-SAT prefers int coefficients)
            coeff = int(weight * rag_adj) + milp_bonus
            numeric_coeffs.append(coeff)
            var_list.append(vars_map[(pkg, v)])

    # offset numeric coefficients to be non-negative
    if numeric_coeffs:
        min_coeff = min(numeric_coeffs)
        if min_coeff < 0:
            offset = -min_coeff
            numeric_coeffs = [c + offset for c in numeric_coeffs]

    # Build linear objective from numeric coeffs and vars
    if numeric_coeffs:
        linear_terms = []
        for coeff, var in zip(numeric_coeffs, var_list):
            linear_terms.append(coeff * var)
        model.Minimize(sum(linear_terms))
    else:
        model.Minimize(0)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = 8
    res = solver.Solve(model)
    if res not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    chosen = {}
    for (pkg, v), var in vars_map.items():
        if solver.Value(var) == 1:
            chosen[pkg] = v

    return chosen


def solve_milp(candidates: Dict[str, List[str]], deps_graph: Any, top_level_pins: Dict[str, Optional[str]]) -> Optional[Dict[str, str]]:
    """
    Solve MILP selecting one version per package from candidates to satisfy constraints.
    deps_graph: networkx graph with edges pkg -> dep (edge attribute 'constraint' string)
    Returns dict: {pkg: chosen_version} or None if infeasible or pulp not available.
    """
    try:
        import pulp # type: ignore
    except Exception:
        return None

    # Build variable x_pkg_ver
    prob = pulp.LpProblem('dependency_resolution', pulp.LpMinimize)
    x = {}
    for pkg, vers in candidates.items():
        for v in vers:
            x[(pkg, v)] = pulp.LpVariable(f"x_{pkg}_{v.replace('.', '_')}", cat='Binary')

    # Constraint: exactly one version per package (if candidates exist)
    for pkg, vers in candidates.items():
        if not vers:
            continue
        prob += pulp.lpSum([x[(pkg, v)] for v in vers]) == 1

    # Constraint: incompatible pairs forbidden
    # For each edge parent -> dep, if parent requires dep spec, then for any candidate of dep that doesn't satisfy spec, forbid the combination.
    for parent in deps_graph.nodes():
        for dep in deps_graph.successors(parent):
            constraint = deps_graph.edges[parent, dep].get('constraint')
            if not constraint:
                continue
            dep_cands = candidates.get(dep, [])
            parent_cands = candidates.get(parent, [])
            if not dep_cands or not parent_cands:
                continue
            try:
                specset = SpecifierSet(constraint)
            except Exception:
                specset = None
            for pv in parent_cands:
                for dv in dep_cands:
                    violates = False
                    if specset:
                        try:
                            if Version(dv) not in specset:
                                violates = True
                        except Exception:
                            violates = False
                    # If no specset, assume compatible
                    if violates:
                        prob += x[(parent, pv)] + x[(dep, dv)] <= 1

    # Objective: minimize changes from top-level pins
    obj_terms = []
    for pkg, vers in candidates.items():
        pin = top_level_pins.get(pkg)
        for v in vers:
            weight = 0
            if pin:
                pin_ver = pin[2:] if pin.startswith('==') else pin
                if v != pin_ver:
                    weight = 1
            else:
                weight = 1  # prefer not changing unspecified? still use 1
            obj_terms.append(weight * x[(pkg, v)])
    prob += pulp.lpSum(obj_terms)

    # Solve
    solver = pulp.PULP_CBC_CMD(msg=False)
    res = prob.solve(solver)
    if pulp.LpStatus[res] != 'Optimal' and pulp.LpStatus[res] != 'Feasible':
        return None

    chosen = {}
    for (pkg, v), var in x.items():
        if var.value() == 1:
            chosen[pkg] = v

    return chosen


def final_resolve(conflicts, top_level_deps, auto_fix=False, out_path="requirements_final.txt", solver: str = "ortools", workers: int = 8):
    """Produce final pinned requirements using heuristics + RAG cache + solver (ortools preferred).
    Returns dict with base suggestions, final_changes, final_file.
    """
    base = suggest_fixes(conflicts, top_level_deps, auto_fix=False, out_path=out_path)
    changes = dict(base.get("changes", {}))

    # helper: scoring function for a candidate assignment
    def score_assignment(assign: Dict[str, str], top_map: Dict[str, Optional[str]], rag_weights: Dict[str, Dict[str, float]]):
        score = 0
        for pkg, ver in assign.items():
            pin = top_map.get(pkg)
            if pin:
                pin_ver = pin[2:] if pin.startswith('==') else pin
                if ver != pin_ver:
                    score += 100
            else:
                score += 100
            # apply RAG multiplier (lower is better)
            if rag_weights and rag_weights.get(pkg) and rag_weights[pkg].get(ver) is not None:
                score = score * rag_weights[pkg][ver]
        return score

    # build dependency graph from conflicts
    try:
        import networkx as nx
    except Exception:
        nx = None

    G = None
    if nx:
        G = nx.DiGraph()
        for name, pin in top_level_deps:
            G.add_node(name)
        for c in conflicts:
            if 'package' in c:
                target = c['package']
                for inc in c.get('incoming_constraints', []):
                    parent = inc.get('from')
                    constraint = inc.get('constraint')
                    if parent and target:
                        G.add_node(parent)
                        G.add_node(target)
                        G.add_edge(parent, target, constraint=constraint)
            elif 'parent' in c:
                parts = c['requires'].split(None, 1)
                if parts:
                    target = parts[0]
                    constraint = parts[1] if len(parts) > 1 else None
                    parent = c.get('parent')
                    if parent and target:
                        G.add_node(parent)
                        G.add_node(target)
                        G.add_edge(parent, target, constraint=constraint)

    final_changes = dict(changes)
    if G is not None:
        packages = list(G.nodes())
        top_map = {name.lower(): pin for name, pin in top_level_deps}
        candidates = _collect_candidates(packages, top_map, max_per_package=5, workers=workers)

        # Compute RAG weights from history: values in (0.1 .. 1.0] where lower reduces penalty
        rag_history = _utils.read_conflict_history(200)
        rag_weights = {}
        for pkg, vers in candidates.items():
            rag_weights[pkg] = {}
            for v in vers:
                # default multiplier 1.0 (no change)
                rag_weights[pkg][v] = 1.0
        # count occurrences of pkg/version in history as a proxy for stability
        hist_counts = {}
        for entry in rag_history:
            c = entry.get('conflict', {})
            pkg = c.get('package')
            # try to glean chosen version from incoming constraints or top_level_pin
            top_pin = c.get('top_level_pin')
            if pkg and top_pin:
                v = top_pin[2:] if isinstance(top_pin, str) and top_pin.startswith('==') else top_pin
                if v:
                    hist_counts.setdefault((pkg, v), 0)
                    hist_counts[(pkg, v)] += 1
        # convert counts to multiplier in (0.5 .. 1.0]
        for (pkg, v), cnt in hist_counts.items():
            if pkg in rag_weights and v in rag_weights[pkg]:
                # more occurrences -> lower multiplier (prefer stable)
                rag_weights[pkg][v] = max(0.5, 1.0 - min(cnt, 10) * 0.05)

        # Run MILP and OR-Tools in parallel to save wall-clock time
        from concurrent.futures import ThreadPoolExecutor
        milp_sol = None
        ort_res = None
        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_milp = ex.submit(solve_milp, candidates, G, {k.lower(): v for k, v in top_map.items()})
            fut_ort = ex.submit(solve_ortools, candidates, G, {k.lower(): v for k, v in top_map.items()}, rag_weights, None, 10.0)
            try:
                milp_sol = fut_milp.result(timeout=30)
            except Exception:
                milp_sol = None
            try:
                ort_res = fut_ort.result(timeout=30)
            except Exception:
                ort_res = None

        # If OR-Tools supports weighting (call internal wrapper)
        try:
            ort_res = None
            # call internal with rag and milp influence
            ort_internal = solve_ortools.__wrapped__ if hasattr(solve_ortools, '__wrapped__') else None
        except Exception:
            ort_internal = None

        # Prefer calling the internal implementation with rag_weights and milp preference
        try:
            # _inner is defined in solve_ortools; attempt to call it directly if available
            if hasattr(solve_ortools, '__call__'):
                # our solve_ortools returns the _inner by default; call main inner with params
                ort_res = solve_ortools(candidates, G, {k.lower(): v for k, v in top_map.items()})
        except Exception:
            ort_res = None

        # As a fallback, call solve_ortools via the public API but prefer milp bonus by recomputing
        if ort_res is None:
            ort_res = solve_ortools(candidates, G, {k.lower(): v for k, v in top_map.items()})

        # Score and pick best solution among ort_res and milp_sol
        chosen = None
        best_score = None
        for sol in (ort_res, milp_sol):
            if not sol:
                continue
            # verify solution satisfies constraints
            feasible = True
            for parent in G.nodes():
                for dep in G.successors(parent):
                    constraint = G.edges[parent, dep].get('constraint')
                    if not constraint:
                        continue
                    dv = sol.get(dep)
                    if not dv:
                        feasible = False
                        break
                    try:
                        specset = SpecifierSet(constraint)
                        if Version(dv) not in specset:
                            feasible = False
                            break
                    except Exception:
                        pass
                if not feasible:
                    break
            if not feasible:
                continue
            sc = score_assignment(sol, top_map, rag_weights)
            if best_score is None or sc < best_score:
                best_score = sc
                chosen = sol

        if chosen:
            final_changes = dict(chosen)

        # Apply compatibility alignment rules for known coupled packages
        def _apply_compat_rules(final_map: Dict[str, str], candidates_map: Dict[str, List[str]]):
            # rules: primary -> [dependents]
            COMPAT_RULES = {
                'torch': ['torchaudio', 'torchvision'],
                'seaborn': ['matplotlib'],
            }
            for primary, dependents in COMPAT_RULES.items():
                if primary not in final_map:
                    continue
                try:
                    pv = Version(final_map[primary])
                except Exception:
                    continue
                target_major = pv.major
                target_minor = pv.minor
                for d in dependents:
                    if d not in candidates_map:
                        continue
                    # try to pick a candidate for d whose major/minor matches primary
                    pick = None
                    for v in reversed(candidates_map[d]):
                        try:
                            vv = Version(v)
                        except Exception:
                            continue
                        if vv.major == target_major and vv.minor == target_minor:
                            pick = v
                            break
                    # fallback: choose latest candidate
                    if not pick and candidates_map[d]:
                        pick = candidates_map[d][-1]
                    if pick:
                        final_map[d] = pick

        _apply_compat_rules(final_changes, candidates)

    # ensure compatibility per aggregated spec (same as before)
    for c in conflicts:
        target = None
        if 'parent' in c:
            parts = c['requires'].split(None, 1)
            if parts:
                target = parts[0]
                combined_spec = parts[1] if len(parts) > 1 else ''
        elif 'package' in c:
            target = c['package']
            specs = [inc.get('constraint') for inc in c.get('incoming_constraints', []) if inc.get('constraint')]
            combined_spec = ",".join(specs) if specs else ''
        else:
            continue

        meta = _utils.get_cached_pypi(target) or resolver.fetch_pypi_info(target)
        if meta:
            _utils.cache_pypi_info(target, meta)
        if not meta:
            continue
        try:
            specset = SpecifierSet(combined_spec) if combined_spec else SpecifierSet()
        except Exception:
            specset = SpecifierSet()

        candidates_list = resolver.find_compatible_versions(meta, specset)
        if candidates_list:
            chosen_version = candidates_list[-1]
            if target in final_changes:
                try:
                    if Version(final_changes[target]) not in specset:
                        final_changes[target] = chosen_version
                except Exception:
                    final_changes[target] = chosen_version
            else:
                final_changes[target] = chosen_version

    # write final file
    final_file = None
    if final_changes:
        try:
            lines = []
            for name, pin in top_level_deps:
                if name in final_changes or name.lower() in {k.lower(): v for k, v in final_changes.items()}:
                    v = final_changes.get(name) or final_changes.get(name.lower())
                    lines.append(f"{name}=={v}\n")
                else:
                    lines.append(f"{name}=={pin if pin and pin.startswith('==') else (pin or '')}\n" if pin else f"{name}\n")
            with open(out_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            final_file = out_path
        except Exception:
            final_file = None

    return {"base": base, "final_changes": final_changes, "final_file": final_file}



