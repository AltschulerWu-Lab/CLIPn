from setuptools import setup, find_packages

setup(
    name="clipn",
    version="0.1.0",
    author="Feng Bao",
    author_email="fbao0110@gmail.com",
    description="Contrastive integration of multiple phenotypic screen datasets",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mypackage",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "torch",
        "numpy",
        "tqdm"
    ],
)