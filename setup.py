import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

README = (HERE / 'README.md').read_text()

setup(
    name='PyRKHSstats',
    version='2.1.0',
    description='A Python package for kernel methods in Statistics/ML.',
    long_description=README,
    long_description_content_type='text/markdown',
    url='https://github.com/Black-Swan-ICL/PyRKHSstats',
    author='K. M-H',
    author_email='kmh.pro@protonmail.com',
    license='GNU General Public License v3.0',
    classifiers=[
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Financial and Insurance Industry',
    ],
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'scikit-learn',
        'GPy',
        'pyyaml'
    ],
    python_requires=">=3.8"
)
