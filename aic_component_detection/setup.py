import os
from glob import glob

from setuptools import find_packages, setup

package_name = "aic_component_detection"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob(os.path.join("launch", "*.launch.xml"))),
        (os.path.join("share", package_name, "config"), glob(os.path.join("config", "*.yaml"))),
        (os.path.join("share", package_name, "model"), glob(os.path.join("model", "*.pt"))),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="User",
    maintainer_email="user@example.com",
    description="SC port and NIC port component detection for the AI Challenge qualifying phase",
    license="Apache-2.0",
    extras_require={
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            "component_detection = aic_component_detection.component_detection:main",
            "zone_projection = aic_component_detection.zone_projection:main",
            "simple_port_3d = aic_component_detection.simple_port_3d:main",
        ],
    },
)
