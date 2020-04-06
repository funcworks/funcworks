from setuptools import setup

if __name__ == '__main__':
    import versioneer
    from funcworks import __version__, DOWNLOAD_URL

    cmdclass = versioneer.get_cmdclass()

    setup(
        name='funcworks',
        version=__version__,
        cmdclass=cmdclass,
        download_url=DOWNLOAD_URL
    )
