#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" A Monte Carlo to Hessian conversion tool """

__author__ = 'Stefano Carrazza'
__license__ = 'GPL'
__version__ = '1.0.0'
__email__ = 'stefano.carrazza@mi.infn.it'

import numpy as np
import lhapdf
import fastcache

DEFAULT_Q = 1.0

class LocalPDF:
    """ A simple class for PDF manipulation """
    def __init__(self, pdf_name, xgrid, fl, Q):
        self.pdf_name = pdf_name
        self.pdf = lhapdf.mkPDFs(pdf_name)
        self.n_rep = len(self.pdf)-1
        self.fl = fl
        self.xgrid = xgrid
        self.xfxQ = np.zeros(shape=(self.n_rep, fl.n, xgrid.n))

        # load replicas at Q
        self.setQ(Q)

    def setQ(self,Q):

        self.Q = Q
        # precomputing values
        for r in range(0,self.n_rep):
            for f in range(self.fl.n):
                for ix in range(self.xgrid.n):
                    self.xfxQ[r, f, ix] = self.pdf[r+1].xfxQ(self.fl.id[f], self.xgrid.x[ix], Q)

        # precomputing averages
        self.f0 = np.zeros(shape=(self.fl.n, self.xgrid.n))
        for f in range(self.fl.n):
            for ix in range(self.xgrid.n):
                self.f0[f, ix] = self.pdf[0].xfxQ(self.fl.id[f], self.xgrid.x[ix], Q)

        # compute std dev
        self.std = np.std(self.xfxQ, axis=0, ddof=1)

 
class XGrid:
    """ The x grid points used by the test """
    def __init__(self, xminlog=1e-5, xminlin=1e-1, xmax=1, nplog=50, nplin=50):
        self.x = np.append(np.logspace(np.log10(xminlog), np.log10(xminlin), 
                                       num=nplog, endpoint=False),
                           np.linspace(xminlin, xmax, num=nplin, endpoint=False)
                          )
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
