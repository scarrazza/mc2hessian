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

## Usage

```Shell
$ ./mc2hessian.py --help
   usage: ./mc2hessian [PDF LHAPDF set] [Number of replicas] [Input energy] [OPTIONAL: replicas for base file]

```

The output of this script is a hessian set of PDF in the LHAPDF6 format.

## Contact Information

Maintainer: Stefano Carrazza (stefano.carrazza@mi.infn.it)

Homepage: