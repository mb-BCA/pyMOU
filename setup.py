"""setup.py"""

from setuptools import setup, find_packages

with open("requirements.txt") as reqs_file:
    REQS = [line.rstrip() for line in reqs_file.readlines() if line[0] not in ['\n', '-', '#']]

setup(
    name = 'pymou',
    description = 'A library for simulation and connectivity estimation of multivariate Ornstein-Uhlenbeck (MOU) process in Python / NumPy.',
    url = 'https://github.com/mb-BCA/pyMOU',
    version = '1.0.dev0',
    license = 'Apache-2.0',

    author = 'Matthieu Gilson, Andrea Insabato, Gorka Zamora-López',
    author_email = 'galib@zamora-lopez.xyz',

    install_requires = REQS,
    packages = find_packages(exclude=['doc', '*tests*']),
    scripts = [],
    include_package_data = True,

    keywords = 'network dynamic system, Ornstein-Uhlenbeck process, connectivity estimation',
    classifiers = [
        'Development Status :: 1 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Software Development :: Libraries :: Python Modules'
        ]

    )
