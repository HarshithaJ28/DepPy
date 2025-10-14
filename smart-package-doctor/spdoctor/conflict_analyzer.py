import requests
from packaging.requirements import Requirement
from packaging.version import Version, InvalidVersion
from rich.console import Console

console = Console()

def check_conflicts(dependencies):
    """
    Fetches dependency metadata from PyPI and detects version conflicts.
    dependencies: list of tuples (name, version)
    """
    conflicts = []
    pinned_versions = {name.lower(): version for name, version in dependencies if version}

    for name, version in dependencies:
        name_lower = name.lower()
        try:
            # Fetch metadata from PyPI JSON API
            url = f"https://pypi.org/pypi/{name_lower}/json"
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                console.print(f"[yellow]⚠️ Could not fetch metadata for {name}[/yellow]")
                continue

            data = resp.json()
            info = data.get("info", {})
            requires_dist = info.get("requires_dist") or []

            for req_str in requires_dist:
                try:
                    req = Requirement(req_str)
                except Exception:
                    continue

                dep_name = req.name.lower()
                if dep_name in pinned_versions:
                    pinned = pinned_versions[dep_name]
                    if pinned == "latest":
                        continue

                    try:
                        pinned_ver = Version(pinned)
                    except InvalidVersion:
                        continue

                    # Check if pinned version satisfies the requirement
                    if pinned_ver not in req.specifier:
                        conflicts.append({
                            "parent": name,
                            "requires": f"{dep_name} {req.specifier}",
                            "found": pinned
                        })
        except Exception as e:
            console.print(f"[red]Error checking {name}: {e}[/red]")

    return conflicts


def report_conflicts(conflicts):
    if not conflicts:
        console.print("[bold green]✅ No conflicts detected based on PyPI metadata.[/bold green]")
        return

    # Normalize incoming conflicts into list of {parent, requires, found}
    normalized = []
    seen = set()

    for c in conflicts:
        # Analyzer-style entry
        if isinstance(c, dict) and 'parent' in c and 'requires' in c and 'found' in c:
            key = (c.get('parent'), c.get('requires'), c.get('found'))
            if key in seen:
                continue
            seen.add(key)
            normalized.append({'parent': c.get('parent'), 'requires': c.get('requires'), 'found': c.get('found')})
        # Resolver-aggregated style: {package, incoming_constraints: [{from, constraint}], top_level_pin}
        elif isinstance(c, dict) and 'package' in c:
            pkg = c.get('package')
            pin = c.get('top_level_pin')
            incoming = c.get('incoming_constraints') or []
            for inc in incoming:
                parent = inc.get('from')
                constraint = inc.get('constraint') or ''
                requires = f"{pkg} {constraint}".strip()
                key = (parent, requires, str(pin) if pin is not None else None)
                if key in seen:
                    continue
                seen.add(key)
                normalized.append({'parent': parent, 'requires': requires, 'found': pin})
        else:
            # Unknown shape: try to stringify
            try:
                s = str(c)
                key = (s, s, s)
                if key in seen:
                    continue
                seen.add(key)
                normalized.append({'parent': 'unknown', 'requires': s, 'found': 'unknown'})
            except Exception:
                continue

    if not normalized:
        console.print("[bold green]✅ No conflicts detected based on PyPI metadata.[/bold green]")
        return

    console.print("[bold red]🚨 Conflicts detected![/bold red]")
    for c in normalized:
        console.print(f"• {c['parent']} requires {c['requires']}, but found {c['found']}")
