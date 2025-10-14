import os
import json
import pytest
from packaging.version import Version

from spdoctor import ai_suggester, utils


def make_simple_graph():
    # graph: A -> B requires B>=2.0
    try:
        import networkx as nx
    except Exception:
        pytest.skip("networkx not available")
    G = nx.DiGraph()
    G.add_node('A')
    G.add_node('B')
    G.add_edge('A', 'B', constraint='>=2.0')
    return G


def test_rag_cache_roundtrip(tmp_path):
    pkg = 'examplepkg'
    meta = {'releases': {'1.0.0': {}, '2.0.0': {}}}
    # use temp cache path
    old_dir = os.getcwd()
    try:
        os.chdir(tmp_path)
        utils.cache_pypi_info(pkg, meta)
        loaded = utils.get_cached_pypi(pkg)
        assert loaded is not None
        assert 'releases' in loaded
    finally:
        os.chdir(old_dir)


def test_solvers_return_assignment():
    G = make_simple_graph()
    candidates = {'A': ['1.0.0'], 'B': ['1.0.0', '2.0.0']}
    top = {'a': None, 'b': None}

    # try OR-Tools first
    ort = ai_suggester.solve_ortools(candidates, G, top)
    if ort is not None:
        assert isinstance(ort, dict)
        assert 'A' in ort and 'B' in ort
        # B must be 2.0.0 to satisfy A's constraint
        assert Version(ort['B']) >= Version('2.0.0')
        return

    # else try pulp fallback
    milp = ai_suggester.solve_milp(candidates, G, top)
    assert milp is not None
    assert 'A' in milp and 'B' in milp
    assert Version(milp['B']) >= Version('2.0.0')
