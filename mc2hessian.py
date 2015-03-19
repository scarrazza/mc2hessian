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
import lhapdf
import multiprocessing
from numba import jit
from joblib import Parallel, delayed

DEFAULT_Q = 1.0
DEFAULT_EPSILON = 100


class LocalPDF:
    """ A simple class for PDF manipulation """
    def __init__(self, pdf_name, nrep, xgrid, fl, Q, eps=DEFAULT_EPSILON):
        self.pdf = lhapdf.mkPDFs(pdf_name)
        self.n = nrep
        self.n_rep = len(self.pdf)-1
        self.fl = fl
        self.xgrid = xgrid
        self.Q = Q
        self.xfxQ = numpy.zeros(shape=(self.n_rep, fl.n, xgrid.n))
        self.base = numpy.zeros(shape=self.n_rep, dtype=numpy.int64)
        self.fin = numpy.zeros(shape=self.n, dtype=numpy.int64)

        for i in range(self.n_rep): self.base[i] = i+1
        for i in range(self.n): self.fin[i] = i+1

        # precomputing values
        for r in range(self.n_rep):
            for f in range(fl.n):
                for ix in range(xgrid.n):
                    self.xfxQ[r, f, ix] = self.pdf[self.base[r]].xfxQ(fl.id[f], xgrid.x[ix], Q)

        # precomputing averages
        self.f0 = numpy.zeros(shape=(fl.n, xgrid.n))
        for f in range(fl.n):
            for ix in range(xgrid.n):
                self.f0[f, ix] = self.pdf[0].xfxQ(fl.id[f], xgrid.x[ix], Q)

        # compute std dev
        self.std = numpy.std(self.xfxQ, axis=0, ddof=1)

        # lower, upper 68cl
        self.std68 = numpy.zeros(shape=(fl.n, xgrid.n))
        for f in range(fl.n):
            low, up = get_limits(self.xfxQ[:,f,:], self.f0[f,:])
            self.std68[f] = (up-low)/2.0

        # maximum difference between std vs 68cl. -> create pandas array
        self.mask = numpy.array([ abs(1 - self.std[f,:]/self.std68[f,:]) <= eps for f in range(fl.n)])
        print " [Info] Keeping ", numpy.count_nonzero(self.mask), "nf*nx using (1-std/68cl) <= eps =", eps

    @jit
    def fill_cov(self, nf, nx, xfxQ, f0, mask):
        n = len(xfxQ)
        cov = numpy.zeros(shape=(nf*nx,nf*nx))
        for fi in range(nf):
            for fj in range(nf):
                for ix in range(nx):
                    for jx in range(nx):
                        if mask[fi, ix] and mask[fj, jx]:
                            i = nx*fi+ix
                            j = nx*fj+jx
                            for r in range(n):
                                cov[i, j] += (xfxQ[r, fi, ix] - f0[fi,ix])*(xfxQ[r, fj, jx] - f0[fj,jx])
        return cov/(n-1.0)

    def pdfcovmat(self):
        """ Build PDF covariance matrices """
        cov = self.fill_cov(self.fl.n, self.xgrid.n, self.xfxQ, self.f0, self.mask)
        return cov

    @jit
    def rebase(self, basis):
        fin = numpy.sort(basis)
        ind = 0
        negative = numpy.zeros(shape=(self.n_rep-self.n), dtype=numpy.int64)
        for i in range(self.n_rep):
            it = False
            for j in fin:
                if j == i+1: it = True
            if it == False:
                negative[ind] = i+1
                ind+=1
        self.fin = fin
        self.base = numpy.append(fin, negative)

        # precomputing values
        for r in range(self.n_rep):
            for f in range(self.fl.n):
                for ix in range(self.xgrid.n):
                    self.xfxQ[r, f, ix] = self.pdf[self.base[r]].xfxQ(self.fl.id[f], self.xgrid.x[ix], self.Q)

def get_limits(xfxQ, f0):
    reps,l = xfxQ.shape
    d = numpy.abs(xfxQ - f0)
    ind = numpy.argsort(d, axis=0)
    ind68 = 68*reps//100
    sr = xfxQ[ind,numpy.arange(0,l)][:ind68,:]
    up1s = numpy.max(sr,axis=0)
    low1s = numpy.min(sr,axis=0)
    return low1s, up1s

class XGrid:
    """ The x grid points used by the test """
    def __init__(self, xminlog=1e-5, xminlin=1e-1, nplog=25, nplin=25):
        self.x = numpy.append(numpy.logspace(numpy.log10(xminlog), numpy.log10(xminlin), num=nplog, endpoint=False),
                              numpy.linspace(xminlin, 0.9, num=nplin, endpoint=False))
        self.n = len(self.x)

class Flavors:
    """ The flavor container """
    def __init__(self, nf=3):
        self.id = numpy.arange(-nf,nf+1)
        self.n = len(self.id)

def invcov_sqrtinvcov(cov):
    val, vec = numpy.linalg.eigh(cov)
    invval = numpy.zeros(shape=len(val))
    sqrtval = numpy.zeros(shape=len(val))
    for i in range(len(val)):
        if val[i] > 1e-12:
            invval[i] = 1.0/val[i]
            sqrtval[i] = 1.0/val[i]**0.5
        else:
            print " [Warning] Removing eigenvalue", i, val[i]

    invcov = numpy.dot(vec, numpy.diag(invval)).dot(vec.T)
    sqrtinvcov = numpy.dot(vec, numpy.diag(sqrtval)).dot(vec.T)
    return invcov, sqrtinvcov

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
def dumptofile(F, xval, qval, fval, vec, store_xfxQ, rep0):
    """ Compute eigenvector direction """
    for ix in range(len(xval)):
        for iq in range(len(qval)):
            for fi in range(len(fval)):
                for j in range(len(vec)):
                    F[fi, ix, iq] += vec[j]*(store_xfxQ[j, fi, ix, iq] - rep0[fi, ix, iq])
                F[fi, ix, iq] += rep0[fi, ix, iq]

def load_replica(rep, pdf_name):
    """ Extract information from replica file """
    suffix = ""
    if rep < 10:
        suffix = "000" + str(rep)
    elif rep < 100:
        suffix = "00" + str(rep)
    elif rep < 1000:
        suffix = "0" + str(rep)
    else: suffix = str(rep)

    inn = open(pdf_name + "_" + suffix + ".dat", 'rb')
    # extract header
    header = ""
    done = False
    while not done:
        text = inn.readline()
        header += text
        if text.find("---") >= 0: done = True

    if rep != 0:
        header = header.replace('replica', 'error')

    xtext = []
    qtext = []
    ftext = []
    # load first subgrid
    xtext.append(inn.readline())
    qtext.append(inn.readline())
    ftext.append(inn.readline())

    done = False
    while not done:
        text = inn.readline()
        if text.find("---") >=0:
            text = inn.readline()
            if text != "":
                xtext.append(text)
                qtext.append(inn.readline())
                ftext.append(inn.readline())
            else: done = True

    inn.close()

    return header, xtext, qtext, ftext

@jit
def precachepdf(xfxQ, fval, xval, qval):
    """ load pdf values """
    res = numpy.zeros(shape=(len(fval), len(xval), len(qval)))
    for fi in range(len(fval)):
        for xi in range(len(xval)):
            for qi in range(len(qval)):
                res[fi, xi, qi] = xfxQ(fval[fi], xval[xi], qval[qi])
    return res

def parallelrep(i, nrep, pdf, pdf_name, file, path, xs0, qs0, fs0, vec, store_xfxQ, rep0):
    """ write to file using multiple processors """
    suffix = ""
    if i < 10:
        suffix = "000" + str(i)
    elif i < 100:
        suffix = "00" + str(i)
    elif i < 1000:
        suffix = "0" + str(i)
    else: suffix = str(i)

    print " -> Writing replica", i
    header, xs, qs, fs = load_replica(pdf.base[i-1], path + "/" + pdf_name)

    if xs != xs0 or qs != qs0 or fs != fs0:
        print " Different grids for PDF replica. Using replica 0 grid."
        xs = xs0
        qs = qs0
        fs = fs0

    out = open(file + "/" + pdf_name + "_hessian_" + str(nrep) + "_" + suffix + ".dat", 'wb')

    out.write(header)

    for sub in range(len(xs)):
        out.write(xs[sub])
        out.write(qs[sub])
        out.write(fs[sub])

        xval = xs[sub].split()
        qval = qs[sub].split()
        fval = fs[sub].split()
        xval = [float(ii) for ii in xval]
        qval = [float(ii) for ii in qval]
        fval = [int(ii)   for ii in fval]

        F = numpy.zeros(shape=(len(fval), len(xval), len(qval)))
        dumptofile(F, xval, qval, fval, vec[i-1], store_xfxQ[sub], rep0[sub])

        for ix in range(len(xval)):
            for iq in range(len(qval)):
                for fi in range(len(fval)):
                    print >> out, "%14.7E" % F[fi, ix, iq],
                out.write("\n")
        out.write("---\n")
    out.close()

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

def make_grid(pdf, pdf_name, nrep, vec):
    num_cores = multiprocessing.cpu_count()
    path = lhapdf.paths()[0] + "/" + pdf_name
    file = pdf_name + "_hessian_" + str(nrep)
    print "\n- Exporting new grid:", file
    if not os.path.exists(file): os.makedirs(file)

    # print info file
    inn = open(path + "/" + pdf_name + ".info", 'rb')
    out = open(file + "/" + pdf_name + "_hessian_" + str(nrep) + ".info", 'wb')

    for l in inn.readlines():
        if l.find("SetDesc:") >= 0: out.write("SetDesc: \"Hessian " + pdf_name + "_hessian\"\n")
        elif l.find("NumMembers:") >= 0: out.write("NumMembers: " + str(nrep+1) + "\n")
        elif l.find("ErrorType: replicas") >= 0: out.write("ErrorType: symmhessian\n")
        else: out.write(l)
    inn.close()
    out.close()

    # print replica 0
    print " -> Writing replica 0"
    header, xs0, qs0, fs0 = load_replica(0, path + "/" + pdf_name)

    out = open(file + "/" + pdf_name + "_hessian_" + str(nrep) + "_0000.dat", 'wb')
    out.write(header)

    store_xfxQ = []
    rep0 = []
    for sub in range(len(xs0)):
        out.write(xs0[sub])
        out.write(qs0[sub])
        out.write(fs0[sub])

        xval = xs0[sub].split()
        qval = qs0[sub].split()
        fval = fs0[sub].split()
        xval = [float(i) for i in xval]
        qval = [float(i) for i in qval]
        fval = [int(i)   for i in fval]

        # precache PDF grids
        res = numpy.zeros(shape=(nrep, len(fval), len(xval), len(qval)))
        for r in range(nrep):
            res[r] = precachepdf(pdf.pdf[pdf.base[r]].xfxQ, fval, xval, qval)
        store_xfxQ.append(res)

        # compute replica 0
        rep0.append(precachepdf(pdf.pdf[0].xfxQ, fval, xval, qval))

        for ix in range(len(xval)):
            for iq in range(len(qval)):
                for fi in range(len(fval)):
                    print >> out, "%14.7E" % rep0[sub][fi, ix, iq],
                out.write("\n")
        out.write("---\n")
    out.close()
    # printing eigenstates
    Parallel(n_jobs=num_cores)(delayed(parallelrep)(i, nrep, pdf, pdf_name,
             file, path, xs0, qs0, fs0, vec, store_xfxQ, rep0)
             for i in range(1, nrep+1))
    print " [Done]"


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

    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set(context="paper", font="monospace")
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(cov, vmin=-1e-5, vmax=1e-5, linewidths=0, square=True)
    f.tight_layout()
    plt.show()
    """

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

    print "\n- Solving the linear system for", nrep, "parameters using", num_cores, "cores:"
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
        make_grid(pdf, pdf_name, nrep, vec)

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
