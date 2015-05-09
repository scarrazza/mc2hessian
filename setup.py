from __future__ import print_function
from setuptools import setup, Extension, find_packages

setup (name = 'mc2hessian',
       version = '1.0',
       description = "A Monte Carlo to Hessian transformation tool",
       author = 'Stefano Carrazza and Zahari Kassabov',
       author_email = 'stefano.carrazza@mi.infn.it',
       url = 'https://github.com/scarrazza/mc2hessian',
       long_description = "See `mc2hessian --help` for the full list of options",
       scripts = ['scripts/mc2hessian'],
       package_dir = {'': 'src'},
       packages = find_packages('src'),
       zip_safe = False,
       classifiers=[
            'License :: OSI Approved :: BSD License',
            'Operating System :: OS Independent',
            'Programming Language :: Python',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Physics',
            'Programming Language :: Python :: 2',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.4',
            ],
       )
