![alt text](https://github.com/scarrazza/mc2hessian/raw/master/extra/mc2h.png "Logo")

A Monte Carlo to Hessian conversion tool for PDF sets.

## Download

Download the latest [release](https://github.com/scarrazza/mc2hessian/releases) or clone the master development repository by running the following command:

```Shell
$ git clone https://github.com/scarrazza/mc2hessian.git
```

## Installation

mc2hessian requires python2.7, numba, numpy, joblib, the simplest way to install all required packages is using anaconda distribution:

```Shell
$ conda create --name myenv --file requirements.txt
$ source activate myenv
```
[LHAPDF](https://lhapdf.hepforge.org/) needs to be installed and working correctly with the Python environment.

## Usage

```Shell
$./mc2hessian.py --help
usage: mc2hessian.py [-h] [--epsilon EPSILON] [--basisfile BASIS]
                     [--file FILE]
                     [pdf_name] [nrep] [Q]

positional arguments:
  pdf_name           Name of LHAPDF set
  nrep               Number of basis vectors
  Q                  Energy scale.

optional arguments:
  -h, --help         show this help message and exit
  --epsilon EPSILON  Minimum ratio between one sigma and 68% intervals to
                     select point.
  --basisfile BASIS  File that contains the indexes of the basis, one for
                     line.
  --file FILE        YAML file in the format of basisga.py
```

The output of this script is a hessian set of PDF in the LHAPDF6 format.

### Basis selection
We provide 2 extra scripts for the determination of the best basis of replicas for the conversion: basisgen and basisga. We recommend basisga which uses a genetic algorithm to determine the best set of replica, the full computation takes several hours:

```Shell
$ python basisga.py --help
usage: basisga.py [-h] [--epsilon EPSILON] [--max-iters MAX_ITERS]
                  pdf_name nrep Q

positional arguments:
  pdf_name              Name of LHAPDF set
  nrep                  Number of basis vectors
  Q                     Energy scale.

optional arguments:
  -h, --help            show this help message and exit
  --epsilon EPSILON     Minimum ratio between one sigma and 68% intervals to
                        select point.
  --max-iters MAX_ITERS
```

This program outputs log files with the std.dev. estimator value and the corresponding replicas.

It also outputs a YAML file containing the final results. This can be passed to `mc2hessian.py`  using the `--file` option.

## Contact Information

Maintainer: Stefano Carrazza (stefano.carrazza@mi.infn.it)
