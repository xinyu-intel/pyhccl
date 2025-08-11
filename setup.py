import setuptools


packages = [
    'pyhccl',
    'pyhccl.utils',
]

setuptools.setup(
    name='pyhccl',
    version='0.0.1',
    description="pyhccl - python bindings for Habana Communication library",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/xinyu-intel/pyhccl",
    packages=packages,
    install_requires=[
        'torch'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
 )
