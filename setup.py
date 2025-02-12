from setuptools import setup, find_packages
setup(
    name="clustanom",
    version="0.1.0",
    author="Teoman Berkay Ayaz",
    author_email="your_email@example.com",
    description="ClustAnom is a Scikit-Learn compatible, clustering based anomaly detection library.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/dekaottoman/clustanom",
    packages=find_packages(),
    install_requires=[
        "scikit-learn",
        "numpy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)

