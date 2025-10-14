import os
import pytest
from spdoctor.parser import parse_dependencies

# --- Test files content ---
requirements_content = """
requests==2.31.0
numpy>=1.25.0
pandas
"""

pyproject_content = """
[tool.poetry.dependencies]
python = "^3.10"
requests = "^2.28.0"
flask = "^2.0"
"""

pipfile_content = """
[packages]
requests = "==2.31.0"
flask = "^2.2"

[dev-packages]
pytest = "^8.0"
"""

# --- Helper to write temporary test files ---
def write_temp_file(filename, content):
    with open(filename, "w") as f:
        f.write(content)

# --- Tests ---
@pytest.fixture
def temp_requirements_file(tmp_path):
    file = tmp_path / "requirements.txt"
    write_temp_file(file, requirements_content)
    return str(file)

@pytest.fixture
def temp_pyproject_file(tmp_path):
    file = tmp_path / "pyproject.toml"
    write_temp_file(file, pyproject_content)
    return str(file)

@pytest.fixture
def temp_pipfile(tmp_path):
    file = tmp_path / "Pipfile"
    write_temp_file(file, pipfile_content)
    return str(file)

def test_requirements_parser(temp_requirements_file):
    deps = parse_dependencies(temp_requirements_file)
    assert ("requests", "2.31.0") in deps
    assert ("numpy", "1.25.0") in deps
    assert ("pandas", None) in deps

def test_pyproject_parser(temp_pyproject_file):
    deps = parse_dependencies(temp_pyproject_file)
    assert ("requests", "^2.28.0") in deps
    assert ("flask", "^2.0") in deps

def test_pipfile_parser(temp_pipfile):
    deps = parse_dependencies(temp_pipfile)
    assert ("requests", "==2.31.0") in deps
    assert ("flask", "^2.2") in deps
    assert ("pytest", "^8.0") in deps
