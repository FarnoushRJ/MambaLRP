import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    required = f.read().splitlines()

setuptools.setup(
    name="MambaLRP",
    version="0.1.0",
    author="Farnoush Rezaei Jafari",
    author_email="farnoushrj@gmail.com",
    description="MambaLRP: Explaining Selective State Space Sequence Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FarnoushRJ/MambaLRP",
    packages=setuptools.find_packages(),
    install_requires=required,
    python_requires=">=3.10",
    include_package_data=True,
    keywords=["machine learning", "explainable ai", "interpretability", "mamba", "xai"],
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
    ],
)
