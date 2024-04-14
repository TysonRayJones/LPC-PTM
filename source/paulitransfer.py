'''
    This file computes Pauli transfer matrices (PTM) from an
    operator's Z-basis superoperator, and computes Pauli 
    transfer "maps" from the PTM. A map is a ~quadratically 
    sparser form of the PTM storing only the non-zero weights.
    The PTMs and maps are computed symbolically.

    The explicit runtime computation of a PTM (and map) is 
    necessary only for user operators which are completely 
    general (i.e. specified as matrices or Kraus maps).
    However, we compute PTMs of all canonical gates (e.g.
    Hadamard) for defensive design, and cache them to avoid
    re-computation; this causes an insignificant overhead.
    Some operators howver (like PauliGadget) benefit 
    greatly from bespoke treatment and are not cast to
    PTMs.

    Beware there are many opportunities for small optimisations
    which may prove ultimately unworthwhile since they do not
    address the simulation bottleneck. For example...

        - getPTMFromSuperOperator() can be overloaded for unitary 
          gates with a simpler formulation than the current
          (wherein they are treated as channels). This would avoid
          the need to compute their superoperator.
        - getPTMFromSuperOperator() never needs to evaluate Pauli 
          tensors, and it currently inefficiently produces dense
          matrices. It can instead leverage Pauli decomposition 
          using e.g. Hantzko's or Hamaguchi et al's 2023 algorithms.
        - we don't necessarily need to compute a PTM (matrix) then
          a map; we can sometimes produce the map directly, but we
          lose the debuggable modularity of PTM -> map
        - many 2D matrix loops can be replaced with list
          vectorisable comprehensions
        - many subroutines could leverage sparse matrices

    Note we use numpy.array (instead of sympy.Matrix) to
    make use of the convenience functions like .flatten().
    There is no performance benefit because the numpy array
    element types is necessarily 'object' in order to store 
    Sympy exprs.
'''


import sympy as sp
import numpy as np

from . matrices import getPauliStringMatrix, getMatrixSimplified



'''
    A flag to log when computing a new PTM, for debugging/illustration
'''

_LOG_CACHE = True



'''
    Functions for computing Pauli transfer matrices
'''


def getPTMFromSuperOperator(sup):
    dim = len(sup)
    num = round(np.log2(dim) / 2)

    # prepare norm symbolically to maintain numerical-precision (and be cute)
    norm = 1 / sp.sqrt(dim)

    # inefficiently obtain all num-qubit Pauli string matrices
    strings = [getPauliStringMatrix(i,num) for i in range(dim)]

    # column-flatten Pauli string matrices into 1D Choi-vectors 
    vectors = [stri.transpose().flatten() for stri in strings]

    # prepare an empty PTM matrix
    ptm = np.empty([dim, dim], dtype=object)

    # populate the PTM matrix
    for i in range(dim):
        for j in range(dim):

            # using that Tr(pauli[i] sup pauli[j]) = <bra|sup|ket>
            bra = vectors[i].conjugate()
            ket = vectors[j]
            ptm[i][j] = norm * bra.dot(sup).dot(ket)

    # symbolic PTMs can often be significantly simplified
    ptm = getMatrixSimplified(ptm)
    return ptm



'''
    Functions for computing and caching Pauli transfer maps
'''


def getMapFromPTM(ptm):

    # numerically-safe zero test
    def is_zero(expr):
        eps = 1E-15

        # if expr is not a sympy Expression, then we perform a numerical check
        if not isinstance(expr, sp.Expr):
            return abs(expr) < eps
        
        # otherwise, we remove negligible symbol coefficients...
        expr = expr.replace(lambda e: e.is_Float, lambda f, eps=eps: 0 if abs(f) < eps else f)

        # and query whether the resulting expression simplifies to be numerically negligible
        return sp.N(expr, eps, chop=True) == 0

    # ith elem = list of (j,coeff) produced by ptm upon i-th Pauli
    return [
        {j:v for j,v in enumerate(col) if not is_zero(v)}
        for col in ptm.transpose()]


class PauliTransferMapCache:

    def __init__(self):
        self.maps = {}

    def getMap(self, operator):

        # obtain unique identifier of the operator
        key = operator.getKey()

        # if this operator's map was already computed, return it
        if key in self.maps:
            return self.maps[key]
        
        if _LOG_CACHE:
            print('\t\t\t>>> evaluating and caching PTM of', key)

        # otherwise, compute the map in terms of operator's blank param
        ptm = operator.getPauliTransferMatrix(subbed=False)
        map = getMapFromPTM(ptm)

        # memoize and return the map
        self.maps[key] = map
        return map
    
    def getCoeff(self, mapKey, mapInInd, mapOutInd):
        return self.maps[mapKey][mapInInd][mapOutInd]