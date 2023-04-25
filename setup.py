import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fitransit",
    version="0.0.2",
    author="Zhang Zixin",
    author_email="troyzx@icloud.com",
    description="A fast tool to fit transit light curve.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/troyzx/fitransit",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
