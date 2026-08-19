import os
from glob import glob

from setuptools import find_packages, setup

package_name = "aic_preinsertion_control"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob(os.path.join("launch", "*.launch.xml"))),
        (os.path.join("share", package_name, "config"), glob(os.path.join("config", "*.yaml"))),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="User",
    maintainer_email="user@example.com",
    description="Image-based Cartesian visual servoing to center a detected blob in the camera image",
    license="Apache-2.0",
    extras_require={
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            "preinsertion_control = aic_preinsertion_control.preinsertion_control:main",
        ],
    },
)
