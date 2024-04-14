'''
    This file contains bit-twiddles, making use of
    python's infinite-precion integers to represent
    Pauli-strings as unsigned base-4 numbers.
'''


from math import log2, ceil, prod


def getBit(n, t):
    return (n >> t) & 1


def setBit(n, t, b):
    m = b << t
    return (n & ~(1<<t)) | m


def getSinglePauliFromInd(ind, targ):

    # Pauli is encoded by two adjacent bits
    b0 = getBit(ind, 2*targ)
    b1 = getBit(ind, 2*targ+1)

    # 0=0b00=I, 1=0b01=X, 2=0b10=Y, 3=0b11=Z
    return (b1 << 1) | b0


def getAllPaulisFromInd(ind, numPaulis):

    # effectively returns base-4 digits of ind
    flags = [getSinglePauliFromInd(ind, t) for t in range(numPaulis)]

    # where the rightmost digit is least significant
    return reversed(flags)


def getSubIndFromPauliStringInd(ind, targets):

    # effectively returns specific base-4 digits of ind,
    # ordered according to the order of targets
    sub = 0    
    for i, t in enumerate(targets):
        sub |= getSinglePauliFromInd(ind, t) << (2*i)

    return sub


def setSubIndOfPauliStringInd(initInd, outSubInd, targets):

    outInd = initInd

    # overwrite each targeted Pauli
    for i, targ in enumerate(targets):

        # according to the Paulis in outSubInd 
        pair = getSinglePauliFromInd(outSubInd, i)

        # where each Pauli is constituted by a pair of adjacent bits
        outInd = setBit(outInd, 2*targ,   getBit(pair, 0))
        outInd = setBit(outInd, 2*targ+1, getBit(pair, 1))

    return outInd


def getOperatorWeightOfPauliStringInd(ind):

    # find number of Paulis which are not I=0
    num = getNumPaulisInStringInd(ind)
    paulis = getAllPaulisFromInd(ind, num)
    weight = sum(p > 0 for p in paulis)
    return weight


def getIndOfPauliStringProduct(aInd, bInd):

    # Ia = aI = a, aa = I, for a=I,X,Y,Z
    # XY = YX = Z
    # XZ = ZX = Y
    # YZ = ZY = X
    return aInd ^ bInd


def _getCoeffOfPauliSeqProd(aPaulis, bPaulis):

    # coeff of product of single Paulis
    def getCoeffOfPauliProd(aPauli, bPauli):

        # identity makes no change
        if aPauli == 0 or bPauli == 0:
            return 1

        # paulis are idempotent
        if aPauli == bPauli:
            return 1
        
        # world's laziest Levi-Civita symbol eval 
        coeffs = {
            (1,2): 1j, (2,1):-1j,
            (1,3):-1j, (3,1): 1j,
            (2,3): 1j, (3,2):-1j}
        return coeffs[(aPauli,bPauli)]

    # multiply all one-qubit Pauli prod coeffs
    return prod(
        getCoeffOfPauliProd(a,b) 
        for a,b in zip(aPaulis, bPaulis))


def getNumPaulisInStringInd(ind):
    return ceil(log2(ind)) // 2 if (ind>0) else 1


def getCoeffOfPauliStringProd(aInd, bInd):

    # obtain individual operators, then get coeff
    numPaulis = getNumPaulisInStringInd(max(aInd,bInd))
    aFlags = getAllPaulisFromInd(aInd, numPaulis)
    bFlags = getAllPaulisFromInd(bInd, numPaulis)
    return _getCoeffOfPauliSeqProd(aFlags, bFlags)


def doPauliStringIndsCommute(aInd, bInd):

    # this is a slow, dumb way of determining
    # whether [aInd,bInd]==0; we should replace
    # this with a faster bitwise/parity method since 
    # this function is called in a huge, tight loop

    # short-circuit to avoid log2(0) below
    if (aInd == 0) or (bInd==0):
        return True

    # obtain individual operators (avoid recomp by above func)
    numPaulis = ceil(log2(max(aInd,bInd))) // 2
    aFlags = list(getAllPaulisFromInd(aInd, numPaulis)) # must discard mutable iterator
    bFlags = list(getAllPaulisFromInd(bInd, numPaulis))

    # obtain coefficients 
    abCoeff = _getCoeffOfPauliSeqProd(aFlags, bFlags)
    baCoeff = _getCoeffOfPauliSeqProd(bFlags, aFlags)

    # if coeffs agree, commutator is 0, ergo aInd,bInd commute
    return abCoeff == baCoeff


def getIndOfPauliString(paulis):
   
   # paulis = digits of base-4 unsigned integer
   return sum(
       'IXYZ'.index(char) * 4**i 
       for i, char in enumerate(reversed(paulis)))


def getStringOfPauliInd(ind, numPaulis):
    codes = getAllPaulisFromInd(ind, numPaulis)
    return ''.join('IXYZ'[i] for i in codes)