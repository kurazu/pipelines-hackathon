from setuptools import setup

REQUIRED_PACKAGES = ["click>=8,<9"]

setup(
    name="trainer",
    version="0.2",
    packages=["trainer"],
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    description="My training application.",
)
