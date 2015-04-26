#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" A Monte Carlo to Hessian conversion tool """

__author__ = 'Stefano Carrazza'
__license__ = 'GPL'
__version__ = '1.0.0'
__email__ = 'stefano.carrazza@mi.infn.it'


import argparse

import numpy as np
import fastcache


from common import LocalPDF, XGrid, Flavors, get_limits
from lh import hessian_from_lincomb

DEFAULT_EPSILON = 1000

@fastcache.lru_cache()
def load_pdf(pdf_name, Q):
    fl = Flavors()
    xgrid = XGrid()
    pdf = LocalPDF(pdf_name, xgrid, fl, Q)
    return pdf, fl, xgrid

def refine_relative(nnew, full_diag, part_diag, others):
    mask = np.zeros(others.shape[1], dtype=bool)
    #index = np.arange(others.shape[1])
    for _ in range(nnew):
        remaining = others[: , ~mask]
        worst = np.argmin(part_diag/full_diag)
        best_eig = np.argmax(remaining[worst,:])
        ind = (np.where(~mask)[0])[best_eig]
        
        part_diag += remaining[:, best_eig]
        mask[ind] = True
    return mask
    
def get_diag(U,s):
    Us = np.dot(U, np.diag(s))
    return np.sum(Us**2, axis=1)
    
def compress_X_rel(X, neig):
    U, s, V = np.linalg.svd(X, full_matrices=False)
    norm = np.sqrt(X.shape[1] - 1)
    sn = s/norm
    full_diag = get_diag(U, sn)
    nbig_vects =  neig // 2
    part_diag = get_diag(U[:,:nbig_vects], sn[:nbig_vects])
    
    others = np.dot(U[:,nbig_vects:], np.diag(sn[nbig_vects:]))**2
    nnew = neig - nbig_vects
    mask = np.ones_like(sn, dtype=bool)
    refmask = refine_relative(nnew, full_diag, part_diag, others)
    mask[nbig_vects:] = refmask

    u = U[:,mask]
    vec = V[mask,:].T/norm
    
    cov = np.dot(u, np.dot(np.diag(sn[mask]**2), u.T))
    return vec, cov

def compress_X_abs(X, neig):
    U, s, V = np.linalg.svd(X, full_matrices=False)
    norm = np.sqrt(X.shape[1] - 1)
    sn = s/norm
    u = U[:,:neig]
    vec = V[:neig,:].T/norm
    cov = np.dot(u, np.dot(np.diag(sn[:neig]**2), u.T))
    
    return vec, cov

compress_X = compress_X_rel
    

def main(pdf_name, neig, Q, epsilon=DEFAULT_EPSILON, no_grid=False):

    print "- Monte Carlo 2 Hessian conversion at", Q, "GeV"

    # Loading basic elements
    pdf, fl, xgrid = load_pdf(pdf_name, Q)
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
    # maximum difference between std vs 68cl. -> create pandas array
    print (" [Info] Keeping %d nf*nx using (1-std/68cl) <= eps =%.3f" % 
           (np.count_nonzero(mask), epsilon))

    X = X[mask,:]
     # Step 2: solve the system
    vec, cov = compress_X(X, neig)

    stdh = iter(np.sqrt(np.diag(cov)))

    # Step 3: quick test
    print "\n- Quick test:"
    rmask = mask.reshape(fl.n, xgrid.n)
    est = Norm = 0
    for f in range(fl.n):
        for x in range(xgrid.n):
            if rmask[f,x]:
                t0 = pdf.std[f,x]
                t1 = next(stdh)

                print "1-sigma MonteCarlo (fl,x,sigma):", fl.id[f], xgrid.x[x], t0
                print "1-sigma Hessian    (fl,x,sigma):", fl.id[f], xgrid.x[x], t1
                print "Ratio:", t1/t0

                est += abs(pdf.f0[f,x] * (1-t1/t0))
                Norm += abs(pdf.f0[f,x])
                
    est /= Norm
    print "Estimator:", est

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
