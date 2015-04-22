#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
        self.pdf_name = pdf_name
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

def get_limits(xfxQ, f0):
    reps,l = xfxQ.shape
    d = numpy.abs(xfxQ - f0)
    ind = numpy.argsort(d, axis=0)
    ind68 = 68*reps//100
    sr = xfxQ[ind,numpy.arange(0,l)][:ind68,:]
    up1s = numpy.max(sr,axis=0)
    low1s = numpy.min(sr,axis=0)
    return low1s, up1s
