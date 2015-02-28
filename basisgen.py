#!/usr/bin/env python
""" Generates a basis guess based on linear combination """

__author__ = 'Stefano Carrazza'
__license__ = 'GPL'
__version__ = '1.0.0'
__email__ = 'stefano.carrazza@mi.infn.it'

import sys
import lhapdf
import mc2hessian
import numpy as np

def main(argv):
    # Get input set name
    nrep = []
    pdf_name = ""
    rep_base = ""
    Q = 1.0
    if len(argv) < 3: usage()
    else:
        pdf_name = argv[0]
        v = argv[1].split()
        for i in v:
            nrep.append(int(i))
        Q = float(argv[2])
    print "- Basis generator at", Q, "GeV"
    pdf = lhapdf.mkPDFs(pdf_name)

    # general setup
    nf = 7
    x = np.append(np.logspace(-5, -1, 50, endpoint=False), np.linspace(0.1, 1, 50, endpoint=False))
    nx = len(x)

    # split into blocks of 100 PDFs in order to avoid understimated systems.
    d = 100
    if (len(pdf)-1) < d or (len(pdf)-1) % d != 0:
        print " [Error] modify the subset size d in order to match the size of the prior set."
    p = (len(pdf)-1)/d

    # compute matrices for linear dependence test
    A = np.zeros(shape=(p, nx*nf, d))
    for t in range(p):
        for r in range(d):
            for fl in range(nf):
                for ix in range(len(x)):
                    i = nx*fl + ix
                    A[t, i, r] = pdf[r+1+t*p].xfxQ(fl-6, x[ix], Q)

    # solving the system and saving residuals
    res = np.empty(p*d)
    for t in range(p):
        for r in range(d):
            b = A[t,:,r]
            a = np.delete(A[t], r, axis=1)
            res[r+t*d] = np.linalg.lstsq(a,b)[1]

    sres = np.sort(res)

    for n in nrep:
        eps = sres[len(sres)-1-n]
        print "\n- Selecting", n, ", cutoff for residuals:", eps
        print "- Printing final", len(np.where(res > eps)[0]), "replicas"

        rep = np.where(res > eps)[0]+1
        f = open(pdf_name + "_hessian_" + str(n) + ".dat", 'wb')
        for i in range(len(rep)):
            f.write(str(int(rep[i])) + "\n")
        f.close()

    print " [Done]\n\n Now run the mc2hessian script with the custom basis.\n"

def usage():
    print "usage: ./basisgen.py [PDF LHAPDF set] [Number of replicas or string array] [Input energy]"
    print "output: file with custom basis for the mc2hessian.py script\n"
    exit()

if __name__ == "__main__":
    mc2hessian.splash()
    main(sys.argv[1:])