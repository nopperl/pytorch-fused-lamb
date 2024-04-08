from setuptools import find_packages, setup


setup(
    name="pytorch_fused_lamb",
    version="0.0.1",
    packages=find_packages(),
#    install_requires=["torch>=2.2.0"],
    tests_require=["transformers>4", "datasets>2"],
)

