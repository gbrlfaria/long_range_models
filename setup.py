from setuptools import find_packages, setup

install_requires = [
    "jax>=0.4.14",
    "flax>=0.7.3",
]

examples_requires = [
    "clu>=0.0.9",
    "optax>=0.1.7",
    "tensorflow>=2.13.0",
    "tensorflow_datasets>=4.9.3",
]

setup(
    name="long_range_models",
    version="0.0.1",
    description="Simple Flax implementations of long-range sequence models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=[
        "artificial intelligence",
        "state space models",
        "long range models",
    ],
    author="Gabriel Faria",
    author_email="gabfaria.cs@gmail.com",
    url="https://github.com/gbrlfaria/long_range_models",
    packages=find_packages(),
    install_requires=install_requires,
    extras_require={
        "examples": examples_requires,
    },
)
