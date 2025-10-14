import os
import json
from typing import Optional, Dict, Any

CACHE_DIR = os.path.join(os.getcwd(), 'data')
PYPI_CACHE = os.path.join(CACHE_DIR, 'pypi_cache')
HIST_CACHE = os.path.join(CACHE_DIR, 'conflict_history.json')

os.makedirs(PYPI_CACHE, exist_ok=True)

# simple in-memory cache used for the lifetime of a single CLI run
IN_MEMORY_PYPI_CACHE: Dict[str, Dict[str, Any]] = {}


def _pypi_path(pkg: str) -> str:
    safe = pkg.replace('/', '_').replace('\\', '_')
    return os.path.join(PYPI_CACHE, f"{safe}.json")


def cache_pypi_info(pkg: str, meta: Dict[str, Any]):
    try:
        with open(_pypi_path(pkg), 'w', encoding='utf-8') as f:
            json.dump(meta, f)
        # also populate in-memory cache
        try:
            IN_MEMORY_PYPI_CACHE[pkg] = meta
        except Exception:
            pass
    except Exception:
        pass


def get_cached_pypi(pkg: str) -> Optional[Dict[str, Any]]:
    # check in-memory cache first
    try:
        if pkg in IN_MEMORY_PYPI_CACHE:
            return IN_MEMORY_PYPI_CACHE[pkg]
    except Exception:
        pass

    p = _pypi_path(pkg)
    if not os.path.exists(p):
        return None
    try:
        with open(p, 'r', encoding='utf-8') as f:
            meta = json.load(f)
            # warm in-memory cache
            try:
                IN_MEMORY_PYPI_CACHE[pkg] = meta
            except Exception:
                pass
            return meta
    except Exception:
        return None


def append_conflict_history(entry: Dict[str, Any]):
    try:
        lst = []
        if os.path.exists(HIST_CACHE):
            with open(HIST_CACHE, 'r', encoding='utf-8') as f:
                lst = json.load(f) or []
        lst.insert(0, entry)
        # keep a bounded history
        lst = lst[:200]
        with open(HIST_CACHE, 'w', encoding='utf-8') as f:
            json.dump(lst, f)
    except Exception:
        pass


def read_conflict_history(limit: int = 20):
    try:
        if os.path.exists(HIST_CACHE):
            with open(HIST_CACHE, 'r', encoding='utf-8') as f:
                lst = json.load(f) or []
                return lst[:limit]
    except Exception:
        pass
    return []
