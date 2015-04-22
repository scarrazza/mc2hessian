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

def read_xqf_from_file(f):

    lines = split_sep(f)
    try:
        (xtext, qtext, ftext) = [next(lines) for _ in range(3)]
    except StopIteration:
        return None
    xvals = numpy.fromstring(xtext, sep = " ")
    qvals = numpy.fromstring(qtext, sep = " ")
    fvals = numpy.fromstring(ftext, sep = " ", dtype=numpy.int)
    vals = numpy.fromstring(b''.join(lines), sep= " ")
    return pd.Series(vals, index = pd.MultiIndex.from_product((xvals, qvals, fvals)))


def read_xqf_from_lhapdf(pdf, rep0grids):
    indexes = tuple(rep0grids.index)
    vals = []
    for x in indexes:
        vals += [pdf.xfxQ(x[3],x[1],x[2])]
    return pd.Series(vals, index = rep0grids.index)

def read_all_xqf(f):
    while True:
        result = read_xqf_from_file(f)
        if result is None:
            return
        yield result

#TODO: Make pdf_name the pdf_name instead of path
def load_replica_2(rep, pdf_name, pdf=None, rep0grids=None):

    sys.stdout.write("-> Reading replica from LHAPDF %d \r" % rep)
    sys.stdout.flush()

    suffix = str(rep).zfill(4)
    with open(pdf_name + "_" + suffix + ".dat", 'rb') as inn:
        header = b"".join(split_sep(inn))

        if rep0grids is not None:
            xfqs = read_xqf_from_lhapdf(pdf, rep0grids)
        else:
            xfqs = list(read_all_xqf(inn))
            xfqs = pd.concat(xfqs, keys=range(len(xfqs)))
    return header, xfqs

#Split this to debug easily
def _rep_to_buffer(out, header, subgrids):
    sep = b'---'
    out.write(header)
    out.write(sep)
    for _,g in subgrids.groupby(level=0):
        out.write(b'\n')
        ind = g.index.get_level_values(1)
        numpy.savetxt(out, ind, fmt='%.7E',delimiter=' ', newline=' ')
        out.write(b'\n')
        ind = g.index.get_level_values(2)
        numpy.savetxt(out, ind, fmt='%.7E',delimiter=' ', newline=' ')
        out.write(b'\n')
        #Integer format
        ind = g.index.get_level_values(3)
        numpy.savetxt(out, ind, delimiter=' ', fmt="%d", 
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

def load_all_replicas(pdf, pdf_name):
    rep0headers, rep0grids = load_replica_2(0,pdf_name)
    headers, grids = zip(*[load_replica_2(rep, pdf_name, pdf.pdf[rep], rep0grids) for rep in range(1,pdf.n_rep+1)])
    return [rep0headers] + list(headers), [rep0grids] + list(grids)

def big_matrix(gridlist):
    central_value = gridlist[0]
    X = pd.concat(gridlist[1:], axis=1,
                 keys=range(1,len(gridlist)+1), #avoid confusion with rep0
                 ).subtract(central_value, axis=0)
    if numpy.any(X.isnull()) or X.shape[0] != len(central_value):
        raise ValueError("Incompatible grid specifications")
    return X

def hessian_from_lincomb(pdf, V):

    # preparing output folder
    neig = V.shape[1]

    base = lhapdf.paths()[0] + "/" + pdf.pdf_name + "/" + pdf.pdf_name
    file = pdf.pdf_name + "_hessian_" + str(neig)
    if not os.path.exists(file): os.makedirs(file)

    # copy replica 0
    shutil.copy(base + "_0000.dat", file + "/" + file + "_0000.dat")

    inn = open(base + ".info", 'rb')
    out = open(file + "/" + file + ".info", 'wb')
    for l in inn.readlines():
        if l.find("SetDesc:") >= 0: out.write("SetDesc: \"Hessian " + pdf.pdf_name + "_hessian\"\n")
        elif l.find("NumMembers:") >= 0: out.write("NumMembers: " + str(neig+1) + "\n")
        elif l.find("ErrorType: replicas") >= 0: out.write("ErrorType: symmhessian\n")
        else: out.write(l)
    inn.close()
    out.close()

    headers, grids = load_all_replicas(pdf, base)
    hess_name = file + '/' + file
    result  = (big_matrix(grids).dot(V)).add(grids[0], axis=0, )
    hess_header = b"PdfType: error\nFormat: lhagrid1\n"
    for column in result.columns:
        write_replica(column + 1, hess_name, hess_header, result[column])
