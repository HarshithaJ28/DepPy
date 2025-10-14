from setuptools import setup, find_packages

setup(
    name="smart-package-doctor",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["requirements-parser", "typer", "toml", "rich"],
    entry_points={
        "console_scripts": [
            "spdoctor=spdoctor.cli:app",
        ],
    },
)
