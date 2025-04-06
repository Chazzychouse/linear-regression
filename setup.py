from setuptools import setup, find_packages

setup(
    name="linear_regression",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "matplotlib>=3.4.0",
    ],
    author="Chase Cartwright",
    author_email="ccartwright@abetterway.com",
    description="A simple linear regression implementation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/chazzychouse/linear_regression",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
) 