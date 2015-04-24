#!/usr/bin/env python
""" A Monte Carlo to Hessian conversion tool """

__author__ = 'Stefano Carrazza'
__license__ = 'GPL'
__version__ = '1.0.0'
__email__ = 'stefano.carrazza@mi.infn.it'

import os
import sys
import argparse
import yaml

import numpy
import multiprocessing
from numba import jit
from joblib import Parallel, delayed
from lh import hessian_from_lincomb

from common import LocalPDF, XGrid, Flavors, invcov_sqrtinvcov

DEFAULT_Q = 1.0
DEFAULT_EPSILON = 100

@jit
def chi2(c, nf, nx, rnew, n, xfxQ, f0, invcov):
    """ Function to minimize or test minimization """
    res = 0.0

    for fi in range(nf):
        for fj in range(nf):
            for ix in range(nx):
                i = nx*fi + ix
                for jx in range(nx):
                    j = nx*fj + jx

                    f0a = f0[fi, ix]
                    f0b = f0[fj, jx]

                    a = b = 0
                    for r in range(n):
                        a += c[r]*(xfxQ[r, fi, ix]-f0a)
                        b += c[r]*(xfxQ[r, fj, jx]-f0b)

                    a += f0a - xfxQ[rnew, fi, ix]
                    b += f0b - xfxQ[rnew, fj, jx]

                    res += a*b*invcov[i, j]
    return res

def minintask(i, A, nf, nx, n, xfxQ, f0, invcov, sqrtinvcov):
    """ The minimization routine """
    min = lambda a: chi2(a, nf, nx, i, n, xfxQ, f0, invcov)

    b = numpy.zeros(shape=(nf*nx))    
    for fi in range(nf):
        for ix in range(nx):
            b[nx*fi + ix] = xfxQ[i, fi, ix]-f0[fi, ix]
    b = sqrtinvcov.dot(b)    

    res = numpy.linalg.lstsq(A,b)[0]
    print " -> Replica", i+1, "ERF:", min(res)
    return res

@jit
def comp_hess(nrep, vec, xfxQ, f, x, cv):

    F = numpy.zeros(shape=nrep)
    for i in range(nrep):
        for j in range(nrep):
            F[i] += vec[i][j]*(xfxQ[j, f, x] - cv)
        F[i] += cv

    err = 0
    for i in range(nrep): err += (F[i]-cv)**2

    return err**0.5

def main(pdf_name, nrep, Q, epsilon=DEFAULT_EPSILON, basis=None,
         no_grid=False):
    # Get input set name

    print "- Monte Carlo 2 Hessian conversion at", Q, "GeV"

    # Loading basic elements
    fl = Flavors()
    xgrid = XGrid()
    pdf = LocalPDF(pdf_name, nrep, xgrid, fl, Q, eps=epsilon)
    nx = xgrid.n
    nf = fl.n

    # Step 1: create pdf covmat
    print "\n- Building PDF covariance matrix:"
    cov = pdf.pdfcovmat()
    invcov, sqrtinvcov = invcov_sqrtinvcov(cov)
    print " [Done] "

    # rebase pdfs
    if basis is not None:
        print "\n- Using custom basis"
        if len(basis) > nrep:
            print " [Warning] large custom basis from file"
        print basis[0:nrep]
        pdf.rebase(basis[0:nrep])

    # Step 2: determine the best an for each replica
    an = numpy.zeros(shape=(pdf.n_rep, nrep))
    num_cores = multiprocessing.cpu_count()

    # create matrix to be solved
    A = numpy.zeros(shape=(nf*nx, nrep))
    for fi in range(nf):
        for ix in range(nx):
            ii = nx*fi + ix
            for r in range(nrep):
                A[ii, r] = pdf.xfxQ[r, fi, ix]-pdf.f0[fi, ix]
    A = sqrtinvcov.dot(A)

    print ("\n- Solving a %d*%d linear system with %d cores" % (nrep, pdf.n_rep, 
                                                            num_cores))
    an = Parallel(n_jobs=num_cores)(delayed(minintask)(i,A,nf,nx,nrep,pdf.xfxQ,pdf.f0,invcov,sqrtinvcov) for i in range(pdf.n_rep))
    print " [Done] "

    # Step 3: construct the covariance matrix
    print "\n- Building parameter covariance matrix:"
    acov = numpy.cov(an, rowvar=0)
    ainvcov = numpy.linalg.inv(acov)

    if not numpy.allclose(numpy.dot(acov, ainvcov), numpy.eye(len(acov))):
        print " [Error] Too redundant basis, try to reduce the size of the basis."
        exit()
    print " [Done] "

    # Step 4: solve the system
    eigenvalues, vec = numpy.linalg.eigh(ainvcov)
    for i in range(len(vec)): vec[i] /= eigenvalues[i]**0.5

    # Step 5: quick test
    print "\n- Quick test:"
    prior_cv = pdf.f0
    prior_std = pdf.std
    est = 0
    for f in range(fl.n):
        for x in range(xgrid.n):
            if pdf.mask[f, x]:
                cv = prior_cv[f,x]
                t0 = prior_std[f,x]
                t1 = comp_hess(nrep, vec, pdf.xfxQ, f, x, cv)

                print "1-sigma MonteCarlo (fl,x,sigma):", fl.id[f], xgrid.x[x], t0
                print "1-sigma Hessian    (fl,x,sigma):", fl.id[f], xgrid.x[x], t1
                print "Ratio:", t1/t0

                if t0 != 0: est += abs((t1-t0)/t0)
    print "Estimator:", est

    # Step 6: exporting to LHAPDF
    if not no_grid:
        hessian_from_lincomb(pdf, vec.T)

    #Return estimator for programmatic reading
    return est


def parse_basisfile(basisfile):
    return numpy.loadtxt(basisfile, dtype=numpy.int)

class ParseBasisAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        super(ParseBasisAction, self).__init__(option_strings, dest,
              nargs=None, **kwargs)
        self.dest = 'basis'
    def __call__(self, parser, namespace, values, option_string=None):
        if not values:
            setattr(namespace, self.dest, None)
        else:
            basis = parse_basisfile(values)
            setattr(namespace, self.dest, basis)


argnames = {'pdf_name', 'nrep', 'Q', 'epsilon', 'basis', 'no_grid'}


def parse_file(filename):
    with open(filename) as f:
        d = yaml.load(f)
    return {k:d[k] for k in d if k in argnames}


def splash():
    print "                  ____  _                   _             "
    print "   _ __ ___   ___|___ \| |__   ___  ___ ___(_) __ _ _ __  "
    print "  | '_ ` _ \ / __| __) | '_ \ / _ \/ __/ __| |/ _` | '_ \ "
    print "  | | | | | | (__ / __/| | | |  __/\__ \__ \ | (_| | | | |"
    print "  |_| |_| |_|\___|_____|_| |_|\___||___/___/_|\__,_|_| |_|"
    print "\n  __v" + __version__ + "__ Author: Stefano Carrazza\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('pdf_name', nargs='?',
                        help = "Name of LHAPDF set")
    parser.add_argument('nrep', nargs='?',
                        help="Number of basis vectors", type=int)
    parser.add_argument('Q', type=float,
                        help="Energy scale.", nargs='?')
    parser.add_argument('--epsilon', type=float, default=DEFAULT_EPSILON,
                        help="Minimum ratio between one sigma and "
                        "68%% intervals to select point.")
    parser.add_argument('--basisfile', help="File that contains"
                       " the indexes of the basis, one per line.",
                       action=ParseBasisAction,)
    parser.add_argument('--file', help = "YAML file in the format of "
                        "basisga.py")

    parser.add_argument('--no-grid', action='store_true',
                        help="Do NOT compute and save the LHAPDF grids. "
                        "Output the error function only")

    args = parser.parse_args()
    if args.file:
        if len(sys.argv) > 3:
            parser.error("Too many arguments with the --file option")
        mainargs = parse_file(args.file)
    else:
        if not all((args.pdf_name, args.nrep, args.Q)):
            parser.error("Too few arguments: Either a file is required "
                         "or pdf_name, nrep and Q.")
        mainargs = vars(args)
        mainargs.pop('file')



    splash()
    main(**mainargs)
