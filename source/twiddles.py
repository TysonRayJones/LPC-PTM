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

    # masks for extracting every 2nd bit
    mask0 = 0b01010101010101010101010101010101 # 32 bits = max 16 qubits
    mask1 = mask0 << 1

    # obtain the left (then right) bits of each Pauli pair, in-place
    aBits0, aBits1 = mask0 & aInd, mask1 & aInd
    bBits0, bBits1 = mask0 & bInd, mask1 & bInd

    # shift left bits to align with right bits
    aBits1 >>= 1
    bBits1 >>= 1

    # sets '10' at every Pauli index where individual pairs don't commute
    flags = (aBits0 & bBits1) ^ (aBits1 & bBits0)

    # strings commute if parity of non-commuting pairs is even
    return not (flags.bit_count() % 2)


def getIndOfPauliString(paulis):
   
   # paulis = digits of base-4 unsigned integer
   return sum(
       'IXYZ'.index(char) * 4**i 
       for i, char in enumerate(reversed(paulis)))


def getStringOfPauliInd(ind, numPaulis):
    codes = getAllPaulisFromInd(ind, numPaulis)
    return ''.join('IXYZ'[i] for i in codes)