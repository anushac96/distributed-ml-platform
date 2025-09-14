from setuptools import setup, find_packages

setup(
    name="distributed-ml-platform",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.12.0",
        "pyyaml>=6.0",
        "asyncio",
    ],
)