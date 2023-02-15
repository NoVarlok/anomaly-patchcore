from setuptools import setup, find_packages

setup(
    name="anomaly",
    version="1.0.0",
    description="Anomaly detection based on Patchcore",
    long_description_content_type="text/markdown",

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    python_requires=">=3.7.5",
)
