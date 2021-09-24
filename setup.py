from setuptools import find_packages, setup

REQUIRED_PACKAGES = ["click>=8,<9"]

setup(
    name="trainer",
    version="0.1",
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    description="My training application.",
)
