![alt text](https://github.com/scarrazza/mc2hessian/raw/master/extra/mc2h.png "Logo")

A Monte Carlo to Hessian conversion tool for PDF sets.
For the Genetical Algorithm methodology please visit the [gamethod branch](https://github.com/scarrazza/mc2hessian/tree/gamethod)

## Download

Download the latest [release](https://github.com/scarrazza/mc2hessian/releases) or clone the master development repository by running the following command:

```Shell
$ git clone https://github.com/scarrazza/mc2hessian.git
```

## Installation

This program requires python2.7, numpy, pandas and
LHAPDF6. [LHAPDF](https://lhapdf.hepforge.org/) needs to be installed
and working correctly with the python environment. If you prefer the
simplest way to install all required packages is using the
[Anaconda](https://store.continuum.io/cshop/anaconda/) distribution:

```Shell
$ conda env update -n root -f environment.yml
```

Once all dependencies are satisfied, run:

```Shell
$ python setup.py install
```

Note that the installer script does not check for dependencies. This
will install the `mc2hessian` program and the `mc2hlib` Python library
in the appropriate paths.

## Usage

```Shell
$./mc2hessian --help
usage: mc2hessian [-h] [--epsilon EPSILON] [--no-grid] [pdf_name] [nrep] [Q]

positional arguments:
  pdf_name           Name of LHAPDF set
  nrep               Number of basis vectors
  Q                  Energy scale.

optional arguments:
  -h, --help         show this help message and exit
  --epsilon EPSILON  Minimum ratio between one sigma and 68% intervals to
                     select point.
  --no-grid          Do NOT compute and save the LHAPDF grids. Output the
                     error function only
```

The output of this script is a hessian set of PDF in the LHAPDF6 format.

## Contact Information

Maintainer: Stefano Carrazza (stefano.carrazza@mi.infn.it)
