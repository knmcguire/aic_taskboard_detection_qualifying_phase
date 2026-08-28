from setuptools import find_packages, setup

package_name = "aic_perinsertion_utils"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="User",
    maintainer_email="user@example.com",
    description="Shared perception helpers for the AI Challenge qualifying phase",
    license="Apache-2.0",
    extras_require={
        "test": [
            "pytest",
        ],
    },
)
