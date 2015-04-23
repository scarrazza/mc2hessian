#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" A Monte Carlo to Hessian conversion tool """

__author__ = 'Stefano Carrazza'
__license__ = 'GPL'
__version__ = '1.0.0'
__email__ = 'stefano.carrazza@mi.infn.it'

import numpy as np
import lhapdf

DEFAULT_Q = 1.0

class LocalPDF:
    """ A simple class for PDF manipulation """
    def __init__(self, pdf_name, xgrid, fl, Q):
        self.pdf_name = pdf_name
        self.pdf = lhapdf.mkPDFs(pdf_name)
        self.n_rep = len(self.pdf)-1
        self.fl = fl
        self.xgrid = xgrid
        self.Q = Q
        self.xfxQ = np.zeros(shape=(self.n_rep, fl.n, xgrid.n))

        # precomputing values
        for r in range(0,self.n_rep):
            for f in range(fl.n):
                for ix in range(xgrid.n):
                    self.xfxQ[r, f, ix] = self.pdf[r+1].xfxQ(fl.id[f], xgrid.x[ix], Q)

        # precomputing averages
        self.f0 = np.zeros(shape=(fl.n, xgrid.n))
        for f in range(fl.n):
            for ix in range(xgrid.n):
                self.f0[f, ix] = self.pdf[0].xfxQ(fl.id[f], xgrid.x[ix], Q)

        # compute std dev
        self.std = np.std(self.xfxQ, axis=0, ddof=1)
 
class XGrid:
    """ The x grid points used by the test """
    def __init__(self, xminlog=1e-5, xminlin=1e-1, nplog=25, nplin=25):
        self.x = np.append(np.logspace(np.log10(xminlog), np.log10(xminlin), num=nplog, endpoint=False),
                              np.linspace(xminlin, 0.9, num=nplin, endpoint=False))
        self.n = len(self.x)

class Flavors:
    """ The flavor container """
    def __init__(self, nf=3):
        self.id = np.arange(-nf,nf+1)
        self.n = len(self.id)


from collections import namedtuple
Limits = namedtuple('Limits', ('mean', 'low1s', 'low2s','up1s','up2s'))
def get_limits(ys):
    reps, l = ys.shape
    m = np.mean(ys, axis=0)
    d = np.abs(ys - m)
    ind = np.argsort(d, axis=0)
    ind68 = 68*reps//100
    ind95 = 95*reps//100
    sr = ys[ind,np.arange(0,l)][:ind68,:]
    sr2 = ys[ind,np.arange(0,l)][:ind95,:]
    up1s = np.max(sr,axis=0)
    up2s = np.max(sr2,axis=0)
    low1s = np.min(sr,axis=0)
    low2s = np.min(sr2,axis=0)
    return Limits(m, low1s, low2s, up1s, up2s)
