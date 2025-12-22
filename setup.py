from setuptools import setup, find_packages

setup(
    name="ambedkargpt",
    version="0.1.0",
    author="Jiya Mary Joseph",
    description="SEMRAG-based Retrieval-Augmented Generation system for Ambedkar’s works",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.12",

    package_dir={"": "src"},
    packages=find_packages("src"),

    install_requires=open("requirements.txt").read().splitlines(),

    entry_points={
        "console_scripts": [
            "ambedkargpt=pipeline.ambedkargpt:main",
        ]
    },
)
