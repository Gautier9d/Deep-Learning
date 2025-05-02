from setuptools import setup, find_packages

setup(
    name="deep_dog",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "pytorch-lightning",
        "transformers",
        "pandas",
        "scikit-learn",
        "datasets",
    ],
)
