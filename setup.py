from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="swarm-shield-cuas",
    version="1.0.0",
    author="SWARM-SHIELD Team",
    description="Hierarchical MARL + Neuromorphic Edge Intelligence for Counter-UAS",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/imshivanshutiwari/swarm-shield-cuas",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
