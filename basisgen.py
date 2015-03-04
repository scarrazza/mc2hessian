#!/usr/bin/env python
""" Generates a basis guess based on linear combination """

__author__ = 'Stefano Carrazza'
__license__ = 'GPL'
__version__ = '1.0.0'
__email__ = 'stefano.carrazza@mi.infn.it'

import sys
import numpy
from mc2hessian import XGrid, Flavors, LocalPDF, splash

def main(argv):
    # Get input set name
    nrep = []
    pdf_name = ""
    Q = 1.0
    if len(argv) < 3: usage()
    else:
        pdf_name = argv[0]
        v = argv[1].split()
        for i in v:
            nrep.append(int(i))
        Q = float(argv[2])
    print "- Basis generator at", Q, "GeV."

        # Loading basic elements
    fl = Flavors(3)
    xgrid = XGrid(1e-5, 1e-1, 25, 25)
    pdf = LocalPDF(pdf_name, nrep[0], xgrid, fl, Q, "")
    nx = xgrid.n
    nf = fl.n

    # Step 1: create pdf covmat
    print "\n- Building PDF covariance matrix."
    cov = pdf.pdfcovmat()

    val, vec = numpy.linalg.eigh(cov)
    sqrtval = numpy.zeros(shape=len(val))
    for i in range(len(val)):
        if val[i] > 1e-12:
            sqrtval[i] = 1.0/val[i]**0.5
        else: print " [Warning] Removing eigenvalue", i, val[i]
    sqrtinvcov = numpy.dot(vec, numpy.diag(sqrtval)).dot(vec.T)
    print " [Done] "

    # Step 2: determine the best an for each replica
    d = 5
    p = pdf.n_rep/d
    print "\n- Splitting system in", p, "systems of", d, "replicas."

    # compute matrices for linear dependence test
    A = numpy.zeros(shape=(p, nx*nf, d))
    for t in range(p):
        for r in range(d):
            for fl in range(nf):
                for ix in range(nx):
                    i = nx*fl + ix
                    A[t, i, r] = pdf.xfxQ[r+t*d, fl, ix] - pdf.f0[fl, ix]
        A[t] = sqrtinvcov.dot(A[t])

    # solving the system and saving residuals
    resA = numpy.zeros(shape=(p,d))
    for t in range(p):
        for r in range(d):
            b = A[t, :, r]
            a = numpy.delete(A[t], r, axis=1)
            resA[t, r] = numpy.linalg.lstsq(a,b)[1]
    print " [Done] "

    # select a fraction of replicas of each subset
    print "\n- Building the final system."
    rep = numpy.zeros(shape=p, dtype=numpy.int64)
    B = numpy.zeros(shape=(nf*nx, p))
    for t in range(p):
        s = numpy.argmin(resA[t])
        rep[t] = s+1+t*d
        B[:, t] = A[t, :, s]

    # build the final system and solve
    res = numpy.zeros(p)
    for r in range(p):
        b = B[:, r]
        a = numpy.delete(B, r, axis=1)
        res[r] = numpy.linalg.lstsq(a, b)[1]
    print " [Done] "

    # print replicas
    sres = numpy.sort(res)
    for n in nrep:
        eps = sres[len(sres)-1-n]
        print "\n- Selecting", n, ", cutoff for residuals:", eps

        filename = pdf_name + "_hessian_" + str(n) + ".dat"
        print "- Printing final", filename, "replicas"
        r = numpy.where(res > eps)[0]
        f = open(filename, 'wb')
        for i in range(len(r)):
            f.write(str(rep[r[i]]) + "\n")
        f.close()

    print " [Done]\n\n Now run the mc2hessian script with the custom basis.\n"


def usage():
    print "usage: ./basisgen.py [PDF LHAPDF set] [Number of replicas or string array] [Input energy]"
    print "output: file with custom basis for the mc2hessian.py script\n"
    exit()

if __name__ == "__main__":
    splash()
    main(sys.argv[1:])
