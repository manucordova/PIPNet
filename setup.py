import os
import setuptools

VERSION = "0.0.1"
DESCRIPTION = "PIPNet"

with open("README.md" , "r") as F:
    LONG_DESCRIPTION = F.read()

setuptools.setup(
    name = "pipnet",
    version = VERSION,
    author = "Manuel Cordova",
    author_email = "manuel.cordova@epfl.ch",
    description = DESCRIPTION,
    long_description = LONG_DESCRIPTION,
    long_description_content_type = "text/markdown",
    url = "TBA",
    package_dir = {"": "src"},
    packages = setuptools.find_packages(where="src"),
)