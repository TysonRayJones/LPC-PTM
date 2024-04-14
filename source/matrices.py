'''
    This file defines some matrix operations used by
    operators.py (when building operator matrices) and
    paulitransfer.py (when building PTMs).

    This file uses numpy.array (in lieu of sympy.Matrix) to
    make use of convenience functions like .flatten().
    There is no performance benefit because the numpy array
    element types are necessarily 'object', to store sympy exprs.

    We also use sympy constants and functions (over numerical
    ones) to retain symbolic accuracy and keep symbolic
    expressions pretty (i.e. exclude floating-point terms).
'''


import sympy as sp
import numpy as np
import functools as ft

from . twiddles import getAllPaulisFromInd
from . symbols import getExprSimplified



'''
    Functions for symbolic treatment of matrices
'''


def getMatrixSubstituted(matr, tuples):
    
    # prepare an empty equally-sized matrix
    out = np.empty([len(matr)]*2, dtype=object)

    # copy every element in matr into out...
    for i, row in enumerate(matr):
        for j, elem in enumerate(row):

            # substituting symbols into any scipy expression
            if isinstance(elem, sp.Expr):
                elem = elem.subs(tuples)
            
            out[i][j] = elem

    return out


def getMatrixSimplified(matr):

    # prepare an shallow (but 2D) copy of matr
    out = np.array(matr)

    # simplify every element without modifying matr
    for i, row in enumerate(out):
        for j, elem in enumerate(row):
            out[i][j] = getExprSimplified(elem, symbolic=True, numeric=False)
    
    return out


def getSymbolsInMatrix(matr):

    # collect all unique symbols between all elements
    symbs = set(
        symb for row in matr for elem in row for symb in 
        (elem.free_symbols if isinstance(elem, sp.Expr) else []))

    return list(symbs)



'''
    Functions for computing Z-basis matrices
'''


def getPauliMatrix(code):

    # our 1-qubit homies
    paulis = [
        [[1,0],[0,1]],
        [[0,1],[1,0]],
        [[0,-sp.I],[sp.I,0]],
        [[1,0],[0,-1]]]

    # convert to numpy ndarray
    return np.array(paulis[code])


def getPauliRotationMatrix(code, param):

    # get the gate's Hermitian generator (simply a Pauli matrix)
    gen = sp.Matrix(getPauliMatrix(code))

    # get rotation unitary matrix (simplify to expand into cos and sin)
    matr = sp.exp(- sp.I * param/2 * gen).simplify()

    # cast to numpy array (not essential)
    return np.array(matr)


def getPauliStringMatrix(ind, numPaulis):

    # extract 1-qubit paulis from indexed numPaulis-length basis state
    paulis = getAllPaulisFromInd(ind, numPaulis)
    matrices = [getPauliMatrix(p) for p in paulis]

    # string matrix is the kronecker product of 1-qubit paulis
    matrix = ft.reduce(np.kron, matrices)
    return matrix


def getControlledGateMatrix(matrix, numCtrls):
    
    # prepare diagonal matrix of 1's
    dim = len(matrix) * 2**numCtrls
    out = np.identity(dim, dtype=object) # cannot sparsify :(

    # overwrite bottom-right matrix
    out[-len(matrix):, -len(matrix):] = matrix
    return out


def getSuperOperator(krauses):

    # convert each matrix is a numpy array (ok if already one)
    krauses = map(np.array, krauses)

    # return a 2N-qubit superoperator of the N-qubit Kraus matrices
    return sum(np.kron(m.conjugate(), m) for m in krauses)
