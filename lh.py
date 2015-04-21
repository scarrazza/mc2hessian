# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 22:00:01 2015

@author: zah
"""

import shutil
import pandas as pd
from common import *

def split_sep(f):
    for line in f:
        if line.startswith(b'---'):
            break
        yield line
    
def read_xqf(f):
    lines = split_sep(f)
    try:
        (xtext, qtext, ftext) = [next(lines) for _ in range(3)]
    except StopIteration:
        return None
    xvals = numpy.fromstring(xtext, sep = " ")
    qvals = numpy.fromstring(qtext, sep = " ")
    fvals = numpy.fromstring(ftext, sep = " ", dtype=numpy.int)
    vals = numpy.fromstring(b''.join(lines), sep= " ")
    #vals = vals.reshape((len(xvals), len(qvals), len(fvals)))
    return pd.Series(vals, index = pd.MultiIndex.from_product((xvals, qvals, fvals)))

def read_all_xqf(f):
    while True:
        result = read_xqf(f)
        if result is None:
            return
        yield result

#TODO: Make pdf_name the pdf_name instead of path
def load_replica_2(rep, pdf_name):
    suffix = str(rep).zfill(4)
    with open(pdf_name + "_" + suffix + ".dat", 'rb') as inn:
        header = b"".join(split_sep(inn))
        xfqs = list(read_all_xqf(inn))
        #MultiIndex groupby doesn't give the correct length.
        #Do this only when needed.
        xfqs = pd.concat(xfqs, keys=range(len(xfqs)))
    return header, xfqs

#Split this to debug easily
def _rep_to_buffer(out, header, subgrids):
    sep = b'---'
    out.write(header)
    out.write(sep)
    for _,g in subgrids.groupby(level=0):
        out.write(b'\n')
        for ind in g.index.levels[1:3]:
            numpy.savetxt(out, ind, fmt='%.7E',delimiter=' ', newline=' ')
            out.write(b'\n')
        #Integer format
        numpy.savetxt(out, g.index.levels[3], delimiter=' ', fmt="%d", 
                      newline=' ')
        out.write(b'\n ')
        #Reshape so printing is easy
        reshaped = g.reshape((len(g.groupby(level=1))*len(g.groupby(level=2)), 
                              len(g.groupby(level=3))))
        numpy.savetxt(out, reshaped, delimiter=" ", newline="\n ", fmt='%14.7E')
        out.write(sep)

def write_replica(rep, pdf_name, header, subgrids):
    suffix = str(rep).zfill(4)
    with open(pdf_name + "_" + suffix + ".dat", 'wb') as out:
        _rep_to_buffer(out, header, subgrids)

#TODO: Deduce nrep
def load_all_replicas(pdf):
    headers, grids = zip(*[load_replica_2(rep, pdf) for rep in range(0,pdf.n_rep+1)])
    return headers, grids

def big_matrix(gridlist):
    central_value = gridlist[0]
    X = pd.concat(gridlist[1:], axis=1,
                 keys=range(1,len(gridlist)+1), #avoid confusion with rep0
                 ).subtract(central_value, axis=0)
    if numpy.any(X.isnull()) or X.shape[0] != len(central_value):
        raise ValueError("Incompatible grid specifications")
    return X

def hessian_from_lincomb(pdf, pdf_name, V):

    # preparing output folder
    nrep = V.shape[1]

    base = lhapdf.paths()[0] + "/" + pdf_name + "/" + pdf_name
    file = pdf_name + "_hessian_" + str(nrep)
    if not os.path.exists(file): os.makedirs(file)

    # copy replica 0
    shutil.copy(base + "_0000.dat", file + "/" + file + "_0000.dat")
    # copy info
    shutil.copy(base + ".info", file + "/" + file + ".info")

    inn = open(base + ".info", 'rb')
    out = open(file + "/" + file + ".info", 'wb')
    for l in inn.readlines():
        if l.find("SetDesc:") >= 0: out.write("SetDesc: \"Hessian " + pdf_name + "_hessian\"\n")
        elif l.find("NumMembers:") >= 0: out.write("NumMembers: " + str(nrep+1) + "\n")
        elif l.find("ErrorType: replicas") >= 0: out.write("ErrorType: symmhessian\n")
        else: out.write(l)
    inn.close()
    out.close()

    """
    headers, grids = load_all_replicas(pdf_name)
    hess_name = file + '/' + file + '_%d' % V.shape[1]
    result  = (big_matrix(grids).dot(V)).add(grids[0], axis=0, )
    hess_header = b"PdfType: error\nFormat: lhagrid1\n"
    for column in result.columns:
        write_replica(column + 1, hess_name, hess_header, result[column])
    """