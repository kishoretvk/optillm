from setuptools import setup, find_packages

setup(
    name="optillm",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "networkx",
        "openai",
        "z3-solver",
        "aiohttp",
        "flask",
        "torch",
        "transformers",
        "azure-identity",
        "tiktoken",
        "scikit-learn",
        "litellm",
        "requests",
        "beautifulsoup4",
        "lxml",
        "presidio_analyzer",
        "presidio_anonymizer"
    ],
    author="codelion",
    author_email="codelion@okyasoft.com",
    description="An optimizing inference proxy for LLMs.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/codelion/optillm",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache-2.0 license",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)