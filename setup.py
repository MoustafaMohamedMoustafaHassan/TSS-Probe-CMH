"""
TSS: Transferable Stress Signals
================================
Cross-platform stress detection with interpretable feature channels.

Install (editable / development):
    pip install -e .
    python -m spacy download en_core_web_sm
"""

from setuptools import setup, find_packages

setup(
    name="tss",
    version="1.0.0",
    description="Transferable Stress Signals — interpretable stress detection",
    author="Anonymous (ACL BioNLP 2026 submission)",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "scikit-learn>=1.3",
        "numpy>=1.24",
        "pandas>=2.0",
        "scipy>=1.11",
        "spacy>=3.6",
        "shap>=0.44",
        "statsmodels>=0.14",
        "joblib>=1.3",
        "tqdm>=4.65",
        "openpyxl>=3.1",
    ],
    extras_require={
        "viz": ["matplotlib>=3.7", "seaborn>=0.12"],
    },
    package_data={
        "": ["data/lexicons/*.txt"],
    },
    include_package_data=True,
)
