#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" A Monte Carlo to Hessian conversion tool """

__author__ = 'Stefano Carrazza'
__license__ = 'GPL'
__version__ = '1.0.0'
__email__ = 'stefano.carrazza@mi.infn.it'


import argparse

import numpy as np

from lh import *


def main(pdf_name, neig, Q, epsilon=DEFAULT_EPSILON, no_grid=False):

    print "- Monte Carlo 2 Hessian conversion at", Q, "GeV"

    # Loading basic elements
    fl = Flavors()
    xgrid = XGrid()
    pdf = LocalPDF(pdf_name, neig, xgrid, fl, Q, eps=epsilon)
    nx = xgrid.n
    nf = fl.n

    # Step 1: create pdf covmat
    print "\n- Building PDF covariance matrix:"
    X = (pdf.xfxQ.reshape(pdf.n_rep, nx*nf) - pdf.f0.reshape(nx*nf)).T
    print " [Done] "

    #Epsilon
    l = get_limits(X.T)
    diff = (l.up1s - l.low1s)/2
    std = np.std(X,axis=1)
    mask = (np.abs((diff-std)/diff) < epsilon)
    X = X[mask,:]
    # Step 2: solve the system
    U, s, V = np.linalg.svd(X, full_matrices=False)
    vec = V[:neig,:].T/(pdf.n_rep-1)**0.5

    u = U[:,:neig]
    cov = np.dot(u, np.dot(np.diag(s[:neig]**2/(pdf.n_rep-1)), u.T))
    stdh = iter(np.sqrt(np.diag(cov)))

    # Step 3: quick test
    print "\n- Quick test:"
    rmask = mask.reshape(fl.n, xgrid.n)
    est = 0
    for f in range(fl.n):
        for x in range(xgrid.n):
            if rmask[f,x]:
                t0 = pdf.std[f,x]
                t1 = next(stdh)

                print "1-sigma MonteCarlo (fl,x,sigma):", fl.id[f], xgrid.x[x], t0
                print "1-sigma Hessian    (fl,x,sigma):", fl.id[f], xgrid.x[x], t1
                print "Ratio:", t1/t0

                est += pdf.f0[f,x] * (1-t1/t0)

    print "Estimator:", est/rmask.sum()

    # Step 4: exporting to LHAPDF
    if not no_grid:
        print "\n- Exporting new grid..."
        hessian_from_lincomb(pdf, vec)

    # Return estimator for programmatic reading
    return est

argnames = {'pdf_name', 'neig', 'Q', 'epsilon', 'no_grid'}

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
    parser.add_argument('neig', nargs='?',
                        help="Number of desired eigenvectors", type=int)
    parser.add_argument('Q', type=float,
                        help="Energy scale.", nargs='?')
    parser.add_argument('--epsilon', type=float, default=DEFAULT_EPSILON,
                        help="Minimum ratio between one sigma and "
                        "68%% intervals to select point.")
    parser.add_argument('--no-grid', action='store_true',
                        help="Do NOT compute and save the LHAPDF grids. "
                        "Output the error function only")

    args = parser.parse_args()
    if not all((args.pdf_name, args.neig, args.Q)):
        parser.error("Too few arguments: pdf_name, neig and Q.")
    mainargs = vars(args)

    splash()
    main(**mainargs)
