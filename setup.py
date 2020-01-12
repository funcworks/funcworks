from setuptools import setup, find_packages
setup(
    name = 'funcworks',
    packages = find_packages(),
    entry_points={'console_scripts': [
            'funcworks=funcworks.cli.run:main'
        ]}
)