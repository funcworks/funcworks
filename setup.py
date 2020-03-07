from setuptools import setup, find_packages
setup(
    name='funcworks',
    packages=find_packages(),
    entry_points={'console_scripts': [
        'funcworks=funcworks.cli.run:main'
        ]},
    install_requires=["nipype>=1.4.2", "traits==5.2.0", "pybids>=0.10.1"])
