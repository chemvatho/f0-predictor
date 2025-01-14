from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="f0-predictor",
    version="0.1.0",
    author="chemvatho",
    author_email="your.email@example.com",
    description="A deep learning model for F0 contour prediction with Viterbi smoothing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/chemvatho/f0-predictor",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.2",
        "praat-parselmouth>=0.4.0",
        "matplotlib>=3.3.4",
        "scikit-learn>=0.24.2",
        "scipy>=1.6.0"
    ],
)
