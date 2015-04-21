# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 22:00:01 2015

@author: zah
"""

import numpy
import pandas as pd
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
def load_all_replicas(pdf_name, nrep=1000):
    headers, grids = zip(*[load_replica_2(rep, pdf_name) for rep in range(0,nrep+1)])
    return headers, grids

def big_matrix(gridlist):
    central_value = gridlist[0]
    X = pd.concat(gridlist[1:], axis=1,
                 keys=range(1,len(gridlist)+1), #avoid confusion with rep0
                 ).subtract(central_value, axis=0)
    if numpy.any(X.isnull()) or X.shape[0] != len(central_value):
        raise ValueError("Incompatible grid specifications")
    return X

def hessian_from_lincomb(pdf_name, V):
    #TODO: Handle the case where grids are non-identical
    headers, grids = load_all_replicas(pdf_name)
    #TODO: Write in  proper folder
    hess_name = '1000rep_h/' + '_hessian_%d' % V.shape[1]
    write_replica(0, hess_name, headers[0], grids[0])
    result  = big_matrix(grids).dot(V)
    hess_header = b"PdfType: error\nFormat: lhagrid1\n"
    for column in result.columns:
        #TODO: Write infos.
        write_replica(column + 1, hess_name, hess_header, result[column])
        
if __name__ == '__main__':
    V = numpy.random.rand(1000,100)
    pdf_name = "/usr/local/share/LHAPDF/1000rep/1000rep"
    hessian_from_lincomb(pdf_name, V)
