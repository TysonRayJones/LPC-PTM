'''
    This file defines the quantum operators a user can use
    in their circuits, to simulate their action upon Pauli
    strings. Operators can be completely general, symbolic
    in parameter and in matrix elements, contain many
    parameters, and be specified upon any number of target 
    and control qubits. The parameters and gate symbols 
    are permitted to be complex, or declared as real (or
    probabilities) to invoke analytic simplifications which
    accelerate later evaluation of the surrogate expectation 
    landscape.

    The operators herein are defined with semi-complicated
    subclassing in an effort to minimise code duplication,
    which is inessential to the PTM simulation algorithm,
    and which is irrelevant to this project's implementation
    in a multiple-dispatch object-oriented-free language
    like Julia. However, there are a few crucial elements of
    the operator (sub)class(es) to consider:

    - the getPauliTransferMap() method creates or returns
      a cached copy of an operator's PTM map. This enables 
      ensemble.py to apply operators upon its Pauli strings
      in an abstract operator-agnostic way.

    - similarly, the getMapCoeff() method returns a single
      scalar from the operator's map (or PTM), via indexing
      agnostic to the particular operator. This enables
      ensemble.py to posteriori evaluate the surrogate
      expectation landscape with operator-agnostic logic.

    - canonical, named, parameterised operators (like Rx,
      Damp, Depol) have their matrices herein statically 
      defined in terms of a "Blank symbol" (see symbols.py),
      which are treated as generic, complex scalars. This
      enables users to later give complex values for 
      operator parameters. But operators with restrictedly
      real parameters have analytically simpler elements
      and Pauli transfer maps, and sometimes even fewer
      branches; they're ergo more efficient to simulate, 
      and to posteriori evaluate surrogate landscapes of.
      As such, at operator instantiation, those given
      a user symbol declared to be real (or a probability,
      etc) trigger analytic simplification, once (per cache).

    - PTM computation is expensive, so is never performed
      twice; transfer maps are cached when first evaluated, 
      triggered when a user first instantiates a specific 
      operator. The form of a PTM of a specific operator 
      type (e.g. Rx) depends on the number of control 
      qubits, the number of targets (if variable), and the 
      domain of the parameter(s) (e.g. whether they are 
      permittedly complex, or real, or constrained within 
      [0,1], etc). As such, all these properties must be
      encoded into a unique key which identifies the
      operator in the PTM cache. This is done via method
      getKey()

    We finally note that to retain numerical precision (and
    aesthetic maps), the matrices defined herein avoid
    using floating-point types (like 1.0 or 1j), or numeric
    operations like numpy.sqrt. Instead, we use sympy types
    and functions so that even numerical expressions are 
    retained as symbolic expressions, with numerical
    evaluation deferred until the post-simulation surrogate
    landscape evaluation. We do however use numpy.array
    (in lieu of sympy.array) in order to preserve methods 
    needed by matrices.py (such as .flatten()).
'''


import sympy as sp
import numpy as np

from . symbols import Symb, BlankSymb

from . twiddles import (
    getIndOfPauliString,
    doPauliStringIndsCommute,
    getStringOfPauliInd,
    getCoeffOfPauliStringProd)

from . paulitransfer import (
    getPTMFromSuperOperator,
    getMapFromPTM,
    PauliTransferMapCache)

from . matrices import (
    getMatrixSubstituted, 
    getMatrixSimplified, 
    getSymbolsInMatrix,
    getControlledGateMatrix, 
    getSuperOperator, 
    getPauliMatrix, 
    getPauliStringMatrix, 
    getPauliRotationMatrix)



'''
    Global PTM map cache singleton, used by all Operator 
    instances to avoid repeated calculation of maps
'''

_mapCache = PauliTransferMapCache()



'''
    Operator base classes
'''


class _Operator:
    '''
        The base-class of all gates and channels which defines all
        functions needed to effect the operator upon a Pauli string
    '''

    def __init__(self, targs):

        # permit single-target operators to be specified without a list
        if isinstance(targs, int):
            targs = [targs]

        self.targs = targs

    # will be overriden by gate subclasses (to include ctrls)
    def getQubits(self):
        return self.targs

    # must be overriden by subclass
    def getSuperOperator(self, subbed=None):
        raise NotImplementedError()
    
    # must be overriden by subclass
    def getKey(self):
        raise NotImplementedError()
    
    # the PTM is always computed afresh, because it is memory inefficient to cache
    def getPauliTransferMatrix(self, subbed=True):
        sup = self.getSuperOperator(subbed)
        ptm = getPTMFromSuperOperator(sup)
        return ptm

    # overriden by param ops; the transfer map is only ever computed once, via the cache
    def getPauliTransferMap(self, cached=True, subbed=True):
        if cached:
            map = _mapCache.getMap(self)
        else:
            ptm = self.getPauliTransferMatrix(subbed)
            map = getMapFromPTM(ptm)
        return map
    
    # obtain (potentially symbolic) coefficient from cached map
    def getMapCoeff(self, mapInInd, mapOutInd, paramValues=None):
        return _mapCache.getCoeff(self.getKey(), mapInInd, mapOutInd)
    
    # may be overriden and extended by subclasses
    def __str__(self):
        name = type(self).__name__
        targs = ','.join(map(str, self.targs))
        return f'{name}[{targs}]'


class _Gate(_Operator):
    '''
        A base class for unitaries and other purity-preserving
        operators, which can feature control qubits, and may
        be additionally subclassed to add parameters.
    '''

    # the Gate's base Z-basis matrix, excluding ctrls,
    # which sub-classes will statically override. This
    # may or may not contain a symbolic parameter
    baseMatrix = None

    def __init__(self, targs, ctrls=None):
        super().__init__(targs)

        self.ctrls = ctrls if (ctrls is not None) else []

    def getQubits(self):

        # ctrl qubits are treated as larger significance than targs
        return self.targs + self.ctrls

    def getMatrix(self, subbed=None):

        # doesn't reflect qubit ordering
        return getControlledGateMatrix(self.baseMatrix, len(self.ctrls))
    
    def getSuperOperator(self, subbed=True):

        # calls sub-class' getMatrix() (not necessarily base definition above)
        matrix = self.getMatrix(subbed)

        # superoperator may or may not be substituted with .param
        superop = getSuperOperator([matrix])

        # simplify superoperator based on param domain
        superop = getMatrixSimplified(superop)
        return superop

    def getKey(self):

        # form a key which uniquely identifies the gate type and its
        # dimension, but which is agnostic of parameters qubit indices
        nt = len(self.targs)
        name = type(self).__name__
        key = f'{name}{nt}'

        # only show 'C' prefix when >0 controls (because it's prettier)
        nc = len(self.ctrls)
        if nc > 0:
            key = f'C{nc}{key}'

        return key
    
    def __str__(self):
    
        # get base operator string
        form = str(super().__str__())

        # added controls if present
        if len(self.ctrls) > 0:
            ctrls = ','.join(map(str, self.ctrls))
            form = f'C[{ctrls}]' + form
        
        return form


class _Channel(_Operator):
    '''
        A base class for noise channels which cannot feature control 
        qubits, and may be additionally subclassed to add parameters.
    '''

    # the Channel's base Z-basis Kraus matrices, 
    # to be statically overriden by sub-classes,
    # which may or may not contain a parameter
    krausMatrices = None

    def getKey(self):

        # form a key which uniquely identifies the channel
        # (including dimension, but agnostic of parameters
        #  and qubit indices)
        nt = len(self.targs)
        name = type(self).__name__
        key = f'{name}{nt}'
        return key
    
    def getKrausMatrices(self, subbed=None):

        # un-parameterised channels return kraus matrices unchanged (ignoring subbed)
        return self.krausMatrices
    
    def getSuperOperator(self, subbed=True):

        # calls sub-classes getKrausMatrices() (not base definition above)
        krauses = self.getKrausMatrices(subbed)

        # superoperator may or may not contain subbed .param
        superop = getSuperOperator(krauses)

        # simplify superoperator based on param domain
        superop = getMatrixSimplified(superop)

        return superop



'''
    Parameterised base classes
'''


class _SinglyParameterised:
    '''
        An additional base-class of all operators 
        which contain strictly one parameter, and
        which are defined with a static matrix
        (ergo requiring a "blank symbol")
    '''

    # Singly-parameterised operators have matrix 
    # attributes  specified in terms of this static 
    # blank symbol. At operator instantiation, this 
    # symbol may be substituted/overriden by another 
    # static blank with assumptions matching those of 
    # the user's given symbol. This enables the 
    # operator's matrices and ergo Pauli transfer map 
    # to be later simplified. It is crucial that all
    # blank symbols are static (and ergo that there 
    # are only ever O(1) instances, as determined by
    # the number of unique assumptions used by the
    # user), so that cached PTMs can be reused by 
    # different instances of the same operator class.
    blankParam = BlankSymb()

    def __init__(self, param):

        # accepts only Symb (or Praram or Prob); cannot accept sp.Symbol directly
        assert isinstance(param, Symb)

        # remember the user's symbol, but do not yet substitute it into other attributes
        self.param = param
    
    # must be overriden by subclass, because its action is gate vs channel specific
    def updateBlankParamAssumptions(self):
        raise NotImplementedError()
    
    def getKey(self):

        # combine unparameterised operator's key with one encoding param's assumptions
        base = super().getKey()
        para = self.param.getKey()
        return f'{base}({para})'
    
    def getExprWithSubbedParams(self, expr, values):

        # expr is a sympy expression which will only ever contain
        # BlankSymb instances (not Symb/Param/Prob) because they
        # come from cached PTM maps. Arg values is a dictionary
        # param -> value (not blanks) where value may be a number 
        # type, or another sympy expression. It is gauranteed all
        # blank symbols in expr correspond to those of this operator.
        # However, not all parameters are necessarily in values;
        # unspecified blank parameters are substituted with their
        # user-given (at construction) instance-specific symbols. 

        # replace the blank param with given value, else the operator's param symbol
        repl = values[self.param] if (self.param in values) else self.param
        expr = expr.subs(self.blankParam, repl)

        # if the expression has had all its symbols substituted, numerically evaluate it
        if len(expr.free_symbols) == 0:
            expr = expr.evalf()

        return expr
    
    def getPauliTransferMap(self, cached=True, subbed=True):

        # get the map in terms of the blank param (from cache or freshly computed),
        # because the super's method is unparameterised (and ignores subbed)
        map = super().getPauliTransferMap(cached, subbed=None)

        # optionally replace blank param with user-given param (for aesthetics)...
        if subbed:

            # without modifying the cache's mutable map
            map = [
                {k : self.getExprWithSubbedParams(v,{}) 
                for k,v in outDict.items()}
                for outDict in map]

        return map
    
    def getMapCoeff(self, mapInInd, mapOutInd, paramValues):

        # get the map coeff in terms of the static blank param (from cache),
        # because un-parameterised super ignores paramValues
        coeff = super().getMapCoeff(mapInInd, mapOutInd, paramValues=None)

        # replace blank symbol with param-value (or just the param symbol)
        coeff = self.getExprWithSubbedParams(coeff, paramValues)

        return coeff
    
    def __str__(self):
    
        # combine unparameterised form with (param)
        return f'{super().__str__()}({self.param})'


class _MultiParameterised:
    '''
        An additional base-class of operators 
        which contain any number of parameters
        (including 0 and 1), and are not specified
        with static matrices; they ergo do not
        make use of blank symbols.
    '''

    def __init__(self, params):

        # must only accept Symb/Param/Prob, not sp.Symbol directly
        assert all(isinstance(p, Symb) for p in params)

        self.params = params

    def getExprWithSubbedParams(self, expr, values):
        
        # substitute params which were given explict values (retain others)
        subs = [(p, values[p]) for p in self.params if p in values]
        expr = expr.subs(subs)

        # if the expression has had all its symbols substituted, numerically evaluate it
        if len(expr.free_symbols) == 0:
            expr = expr.evalf()

        return expr
    
    def getMapCoeff(self, mapInInd, mapOutInd, paramValues):

        # get symbolic map coeff (unparameterised super ignores paramValues)
        coeff = super().getMapCoeff(mapInInd, mapOutInd, paramValues=None)

        # replace symbols if given values
        coeff = self.getExprWithSubbedParams(coeff, paramValues)

        return coeff
    
    def __str__(self):
    
        # get unparameterised form
        form = str(super().__str__())

        # added params if present
        if len(self.params) > 0:
            para = ','.join(map(str, self.params))
            form += f'({para})'
        
        return form


class _SingleParamGate(_SinglyParameterised, _Gate):

    def __init__(self, targs, param, ctrls=None):

        # separate constructor args between parent constructors
        _SinglyParameterised.__init__(self, param)
        _Gate.__init__(self, targs, ctrls)

        # use param assumptions to simplify matrices
        self.updateBlankParamAssumptions()

    def updateBlankParamAssumptions(self):

        # get new blank param of identical assumptions to the user-given param
        newBlank = BlankSymb.getBlankWithSameAssumptions(self.param)
        
        # substitute old blank in gate matrix with new one, simplifying subsequent superoperator
        subs = [(self.blankParam, newBlank)]
        self.baseMatrix = getMatrixSubstituted(self.baseMatrix, subs)

        # override the static blank param
        self.blankParam = newBlank

    # override getMatrix to substitute in symbolic parameter
    def getMatrix(self, subbed=True):

        # get symbolic gate matrix, including with control qubits
        matr = super().getMatrix(subbed)

        # optionally replace static dummy param with gate param
        if subbed:
            subs = [(self.blankParam, self.param)]
            matr = getMatrixSubstituted(matr, subs)

        return matr


class _SingleParamChannel(_SinglyParameterised, _Channel):

    def __init__(self, targs, param):

        # separate constructor args between parent constructors
        _SinglyParameterised.__init__(self, param)
        _Channel.__init__(self, targs)

        # use param assumptions to simplify Kraus matrices
        self.updateBlankParamAssumptions()

    def updateBlankParamAssumptions(self):

        # get new blank param of identical assumptions to the user-given param
        newBlank = BlankSymb.getBlankWithSameAssumptions(self.param)

        # substitute old blank in Kraus matrices with new blank, simplifying subsequent superoperator
        subs = [(self.blankParam, newBlank)]
        self.krausMatrices = [getMatrixSubstituted(matr, subs) for matr in self.krausMatrices]
        
        # override the static blank param
        self.blankParam = newBlank

    # override getKrausMatrices to substitute in symbolic parameter
    def getKrausMatrices(self, subbed=True):

        # get blank-param symbolic base matrix (super() -> _Channel)
        krauses = super().getKrausMatrices(subbed)

        # optionally replace static dummy param with gate param (for aesthetics)
        if subbed:
            subs = [(self.blankParam, self.param)]
            krauses = [getMatrixSubstituted(m, subs) for m in krauses]

        return krauses



'''
    Concrete gates
'''


class H(_Gate):
    baseMatrix = 1/sp.sqrt(2) * np.array([
        [1, 1],
        [1, -1]])


class Rx(_SingleParamGate):
    # exp(-i param/2 X)
    baseMatrix = getPauliRotationMatrix(1, _SingleParamGate.blankParam)


class Ry(_SingleParamGate):
    # exp(-i param/2 Y)
    baseMatrix = getPauliRotationMatrix(2, _SingleParamGate.blankParam)


class Rz(_SingleParamGate):
    # exp(-i param/2 Z)
    baseMatrix = getPauliRotationMatrix(3, _SingleParamGate.blankParam)
        

class PauliGadget(_SingleParamGate):

    # exp(-i param/2 (paulis))

    def __init__(self, targs, paulis, param, ctrls=None):

        # see ensemble.py's _applyPauliGadget() method
        if not param.is_real:
            raise NotImplementedError(
                "General (permittedly complex) symbols are not implemented for Pauli gadgets")

        # requires bespoke algorithmic treatment
        if ctrls:
            raise NotImplementedError(
                "Control qubits are not implemented for Pauli gadgets")

        # must specify a Pauli upon every target
        assert len(targs) == len(paulis)

        # convert Pauli list into a big-integer (least significant first)
        self.pauliInd = getIndOfPauliString(paulis[::-1])

        # use param as blankParam, to make inherited coeff substitution work
        self.blankParam = param
        super().__init__(targs, param)

    # no blank params are used because we do not maintain a pre-instance matrix
    def updateBlankParamAssumptions(self):
        pass

    # no key is needed because there is no explicit map to cache
    def getKey(self):
        raise RuntimeError()

    def getGeneratorInd(self):
        return self.pauliInd
    
    def getMapCoeff(self, mapInInd, mapOutFlag, paramValues):
    
        # mapOutFlag flags the map's output state as either mapInInd (0) or prodInd below (1).
        # The former output always has coefficient 1 (no branching) or cos(x) (when branched),
        # while the latter coefficient is always +-sin(x). This is a lazy re-use of the 
        # generic PTM map edge-properties; we could avoid the below commutator recomps by
        # changing the "edge data" saved by ensemble.py, but eh. Post-eval is the cheap part.

        # if state and generator commute, the gadget non-branched with coeff 1
        if doPauliStringIndsCommute(mapInInd, self.pauliInd):
            assert mapOutFlag == 0
            return 1
            
        # otherwise it's cos(x) or +-sin(x); the latter's sign is - im(coeff(mapInInd * pauliInd))
        if mapOutFlag == 0:
            mapCoeff = sp.cos(self.param)
        else: 
            prodCoeff = getCoeffOfPauliStringProd(mapInInd, self.pauliInd)
            mapCoeff = - prodCoeff.imag * sp.sin(self.param)

        # replace param with its value if passed
        mapCoeff = self.getExprWithSubbedParams(mapCoeff, paramValues)
        return mapCoeff
    
    def __str__(self):
        assert len(self.ctrls) == 0

        name = type(self).__name__
        targs = ','.join(map(str, self.targs))
        paulis = getStringOfPauliInd(self.pauliInd, len(self.targs))
        
        return f'{name}[{targs}][{paulis}]({self.param})'
    

class U(_MultiParameterised, _Gate):

    def __init__(self, targs, matrix, ctrls=None):

        # assert dimensional validity (permitting one-target to be an int, else list)
        assert len(matrix) == 2**(1 if isinstance(targs, int) else len(targs))
        assert all(len(matrix) == len(row) for row in matrix)

        # assert matrix is immutable, since we use its id to inform U's cache key
        assert isinstance(matrix, tuple) and all(isinstance(row, tuple) for row in matrix)

        # extract all symbols present in the matrix
        symbs = getSymbolsInMatrix(matrix)

        # construct parents
        _MultiParameterised.__init__(self, symbs)
        _Gate.__init__(self, targs, ctrls)

        # bind the user given matrix
        self.baseMatrix = matrix

    def getKey(self):

        # obtain a generic key of form CnUm
        key = super().getKey()

        # and make it unique to this instance's immutable matrix, 
        # which can be shared by other U instances
        key = f'{key}({id(self.baseMatrix)})'

        # note param assumptions are superfluous and not encoded
        return key



'''
    Concrete channels
'''


class Damp(_SingleParamChannel):
    
    # parameterised Kraus matrices
    krausMatrices = (lambda p : (
        [[1, 0],
         [0, sp.sqrt(1-p)]], 
        [[0, sp.sqrt(p  )],
         [0, 0]]
    )) (_SingleParamChannel.blankParam)


class Deph(_SingleParamChannel):

    krauses_1qb = (lambda p : (
        sp.sqrt(1-p) * getPauliMatrix(0),
        sp.sqrt(p  ) * getPauliMatrix(3),
    )) (_SingleParamChannel.blankParam)

    krauses_2qb = (lambda p : (
        sp.sqrt(1-p) * np.kron(getPauliMatrix(0), getPauliMatrix(0)),
        sp.sqrt(p/3) * np.kron(getPauliMatrix(0), getPauliMatrix(3)),
        sp.sqrt(p/3) * np.kron(getPauliMatrix(3), getPauliMatrix(0)),
        sp.sqrt(p/3) * np.kron(getPauliMatrix(3), getPauliMatrix(3)),
    )) (_SingleParamChannel.blankParam)

    def __init__(self, targs, param):

        # bind 1qb or 2qb matrices, overwriting static krausMatrices 
        # attribute with pointer to one of the static lists
        self.krausMatrices = self.krauses_1qb
        if isinstance(targs, list) and len(targs) == 2:
            self.krausMatrices = self.krauses_2qb

        # then call parent constructors (which may modify .krausMatrices)
        super().__init__(targs, param)


class Depol(_SingleParamChannel):

    krauses_1qb = (lambda p : (
        sp.sqrt(1-p) * getPauliMatrix(0),
        sp.sqrt(p/3) * getPauliMatrix(1),
        sp.sqrt(p/3) * getPauliMatrix(2),
        sp.sqrt(p/3) * getPauliMatrix(3),
    )) (_SingleParamChannel.blankParam)

    krauses_2qb = (lambda p : tuple(
        sp.sqrt((1-p) if (i==0) else (p/15)) *
        getPauliStringMatrix(i, 2)
        for i in range(16)
    )) (_SingleParamChannel.blankParam)

    def __init__(self, targs, param):

        # bind 1qb or 2qb matrices, overwriting static krausMatrices 
        # attribute with pointer to one of the static lists
        self.krausMatrices = self.krauses_1qb
        if isinstance(targs, list) and len(targs) == 2:
            self.krausMatrices = self.krauses_2qb

        # then call parent constructors (which may modify .krausMatrices)
        super().__init__(targs, param)


class Kraus(_MultiParameterised, _Channel):

    def __init__(self, targs, matrices):

        # assert dimensional validity (permitting one-target channels to simply past targ)
        if isinstance(targs, int):
            targs = [targs]
        assert all(len(matr) == 2**len(targs) for matr in matrices)
        assert all(len(matr) == len(row) for matr in matrices for row in matr)

        # assert matrix list is immutable, since we use its id to inform Kraus's cache key
        assert (
            isinstance(matrices, tuple) and 
            all(isinstance(matr, tuple) for matr in matrices) and
            all(isinstance(row, tuple) for matr in matrices for row in matr))

        # extract all symbols present in the matrices
        symbs = list(set(symb for matr in matrices for symb in getSymbolsInMatrix(matr)))

        # construct parents
        _MultiParameterised.__init__(self, symbs)
        _Channel.__init__(self, targs)

        # bind user-given immutable matrices
        self.krausMatrices = matrices

    def getKey(self):

        # obtain a generic key of form KrausN
        key = super().getKey()

        # and make it unique to this instance's immutable Kraus map, 
        # which can be shared by other Kraus instances
        key = f'{key}({id(self.krausMatrices)})'

        # note param assumptions are superfluous and not encoded
        return key