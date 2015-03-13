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

### Basis selection
We provide 2 extra scripts for the determination of the best basis of replicas for the conversion: basisgen and basisga. We recommend basisga which uses a genetic algorithm to determine the best set of replica, the full computation takes several hours:

```Shell
$ ./basisga.py --help
   usage: ./basisga [PDF LHAPDF set] [Number of replicas] [Input energy]
```

This program outputs log files with the std.dev. estimator value and the corresponding replicas.

## Contact Information

Maintainer: Stefano Carrazza (stefano.carrazza@mi.infn.it)
