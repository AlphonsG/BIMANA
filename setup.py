from setuptools import setup

REQUIRED_PACKAGES = (
    'pytest',
    'pytest-cov',
    'natsort',
    'click-web',
    'scipy',
    'opencv-python',
    'pandas',
)

setup(
    name='bimana',
    author='Alphons Gwatimba',
    author_email='0go0vdp95@mozmail.com',
    install_requires=REQUIRED_PACKAGES,
    python_requires='>=3.10',
    entry_points={
        'console_scripts': [
            'bimana = bimana.__main__:main',
        ],
    },
)
