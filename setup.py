import setuptools

VERSION = "1.0.0"
DESCRIPTION = "pipnet"

with open("README.md", "r") as F:
    LONG_DESCRIPTION = F.read()

setuptools.setup(
    name="pipnet",
    version=VERSION,
    author="Manuel Cordova",
    author_email="manuel.cordova@epfl.ch",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/manucordova/PIPNet",
    packages=setuptools.find_packages(where="src"),
    package_dir={"": "src"},
)
