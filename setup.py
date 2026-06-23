from setuptools import setup, find_packages

setup(
    name="trainload",
    version="1.1.0",
    description="Training-load analytics pipeline for endurance athletes",
    packages=find_packages(exclude=["tests"]),
    install_requires=["pandas>=1.1", "numpy>=1.18", "PyYAML>=5.3"],
    python_requires=">=3.8",
)
