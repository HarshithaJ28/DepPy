import os
import toml # type: ignore
from rich.console import Console

console = Console()


def parse_dependencies(file_path: str):
    """
    Detects the dependency file type and parses it.
    Supports: requirements.txt, pyproject.toml, Pipfile
    Returns a list of tuples: (package_name, version)
    """
    if not os.path.isfile(file_path):
        console.print(f"[bold red]Error: File not found:[/bold red] {file_path}")
        return []

    filename = os.path.basename(file_path).lower()

    try:
        # Accept exact names and common variants like `requirements_conflict.txt`.
        if filename == "requirements.txt" or (filename.startswith("requirements") and filename.endswith(".txt")):
            return _parse_requirements(file_path)
        elif filename == "pyproject.toml":
            return _parse_pyproject(file_path)
        elif filename == "pipfile":
            return _parse_pipfile(file_path)
        else:
            console.print(f"[bold red]Unsupported file type:[/bold red] {filename}")
            return []
    except Exception as e:
        console.print(f"[bold red]Failed to parse file:[/bold red] {file_path}")
        console.print(f"[red]{str(e)}[/red]")
        return []


# ----------------- PARSERS ----------------- #

def _parse_requirements(file_path: str):
    """
    Parse requirements.txt
    Format: package==version or package>=version etc.
    """
    deps = []
    try:
        # Use the public API `parse` which yields Requirement objects.
        from requirements import parse # type: ignore
    except ImportError:
        console.print("[bold red]Please install requirements-parser[/bold red]")
        return deps

    with open(file_path, "r") as f:
        try:
            for req in parse(f):
                version = str(req.specs[0][1]) if req.specs else None
                deps.append((req.name, version))
        except Exception:
            # If parsing as a whole fails, fallback to line-by-line best-effort.
            f.seek(0)
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                deps.append((line, None))
    return deps


def _parse_pyproject(file_path: str):
    """
    Parse pyproject.toml (Poetry style)
    """
    deps = []
    data = toml.load(file_path)
    try:
        poetry_deps = data.get("tool", {}).get("poetry", {}).get("dependencies", {})
        for pkg, version in poetry_deps.items():
            # Skip Python version itself
            if pkg.lower() == "python":
                continue
            if isinstance(version, dict):
                version_str = version.get("version")
            else:
                version_str = str(version)
            deps.append((pkg, version_str))
    except Exception:
        console.print(f"[bold red]No dependencies found in pyproject.toml[/bold red]")
    return deps


def _parse_pipfile(file_path: str):
    """
    Parse Pipfile
    """
    deps = []
    data = toml.load(file_path)
    for section in ["packages", "dev-packages"]:
        packages = data.get(section, {})
        for pkg, version in packages.items():
            if isinstance(version, dict):
                version_str = version.get("version")
            else:
                version_str = str(version)
            deps.append((pkg, version_str))
    return deps