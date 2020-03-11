from setuptools import setup, find_packages
from funcworks import __version__
setup(
    name='funcworks',
    version=__version__,
    packages=find_packages(),
    entry_points={'console_scripts': [
        'funcworks=funcworks.cli.run:main'
        ]},
    install_requires=["nipype>=1.4.2", "traits==5.2.0", "pybids>=0.10.1"])
