import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="modelconductor",
    version="0.0.1",
    author="Panu Aho",
    author_email="panu.aho@gmail.com",
    description="Many To Many Co-Simulation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/donkkis/modelconductor",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
