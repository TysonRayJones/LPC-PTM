'''
    This file defines symbols used in user matrices, and in our
    static operator definitions in operators.py, which enable
    deferred evaluation of things like gate parameters and
    Pauli string weights. We define user-facing custom wrapper 
    subclasses (Symb, Param, Prob) of sympy's Symbol class, which 
    respectively impose stronger default assumptions about the 
    symbol's domain (C, R, [0,1]), for user convenience. The domain 
    (or "assumptions") of a symbol greatly affect the simplification
    of its Pauli transfer matrix and map.

    This file also defines BlankSymb which is used by operators.py 
    as a placeholder symbol in static parameterised operator matrices,
    which makes the minimum assumptions (assumes all symbols are complex).

    Note that sympy's Symbol() object actually accepts custom keys to 
    its constructor (like probability=True) which are automatically 
    collected in ._assumptions. However, we'd still have to subclass
    Symbol to set defaults, while losing the .is_probability method. So
    our explicit subclassing here is fine!
'''


import sympy as sp



'''
    User-facing symbolic operator parameters
'''


class Symb(sp.Symbol):
    '''
        A subclass of sp.Symbol enabling users to additionally
        specify a symbol is a "probability" (as appear in 
        decoherence channels) which later enables significant 
        simplification of the symbolic PTMs (asserting 0<=p<=1). 
        By default, a Symbol is assumed to be a complex scalar,
        as would naturally appear as elements in user-specified 
        symbolic operator matrices.

        Note we didn't strictly NEED to subclass; the sp.Symbol()
        constructor accepts custom keys like probability, which we
        could later consult with instance._assumptions['probability'],
        but we'd want a clean .is_probability method anyway. 
    '''

    def __new__(cls, name, probability=False, **kwargs):

        # automatically assert probabilities are >= 0
        if probability:
            assert kwargs['real'] if 'real' in kwargs else True
            assert kwargs['nonnegative'] if 'nonnegative' in kwargs else True
            kwargs['real'] = True
            kwargs['nonnegative'] = True

        # create the symbol, binding 'is_probabiity'
        obj = sp.Symbol.__new__(cls, name, **kwargs)
        obj.is_probability = probability

        # record all explicitly given assumptions for getKey()
        obj._assumps = kwargs
        obj._assumps['probability'] = probability

        return obj
    
    def getKey(self):

        # uniquely identify the assumptions attached to the symbol.
        # note that superflously setting a default value (e.g. complex=True)
        # results in a unique key and the PTM would be re-computed. That's ok.
        return ','.join(f'{key}={val}' for key,val in self._assumps.items())


class Param(Symb):
    '''
        A convenience wrapper of Symb which by default assumes the symbol 
        is real, as relevant to real-parameterised operators like Rx.
    '''

    def __new__(cls, name, **kwargs):

        # assume parameters are real by default, but permit them to be complex
        if 'real' not in kwargs:
            kwargs['real'] = True

        return Symb.__new__(cls, name, **kwargs)


class Prob(Symb):
    '''
        A convenience wrapper of Symb which be default assumes the symbol
        is a probability, as relevant to parameterised channels like Damp.
    '''

    def __new__(cls, name, **kwargs):

        # insist probability is not explicitly specified (forbid complex Prob instances)
        assert 'probability' not in kwargs
        return Symb.__new__(cls, name, probability=True, **kwargs)



'''
    Internal symbol-placeholder
'''


class BlankSymb(sp.Dummy):
    '''
        A wrapper of sympy's Dummy variable which has additional
        attribute is_probability. This blank symbol is used by
        parameterised operator definitions to statically specify 
        a matrix before the user has given a parameter and associated
        assumptions. The default blank symbol makes no simplifying 
        assumptions about its values, leading to complicated matrix
        expressions and PTMs; so at instantiation, an operator's 
        blank symbol is dynamically overriden with one with the
        same assumptions as the user-given symbol. 

        The replacement blank symbol must be one of a fixed, static 
        set so that we always have a handle to the blank symbols
        within a cached PTM, as is needed to later substitute out
        the blanks with user-given parameters or values. Ergo we
        "cache" blank parameters.
    '''

    # static cache of assumptions key -> BlankSymb instances
    symbCache = {}

    def __new__(cls, probability=False, **kwargs):

        # create the Dummy symbol, binding 'is_probabiity'
        obj = sp.Dummy.__new__(cls, 'blank', **kwargs)
        obj.is_probability = probability
        return obj
    
    @classmethod
    def getBlankWithSameAssumptions(cls, symb):

        # we cannot accept sp.Symbol directly because we need ._assumps
        assert isinstance(symb, Symb)

        # get a key which unique identifies the assumptions of the symbol
        key = symb.getKey()

        # return if we've already prepared a blank symbol with these assumptions
        if key in cls.symbCache:
            return cls.symbCache[key]

        # otherwise make and cache a new same-assumption blank symbol
        blank = BlankSymb(**symb._assumps)
        cls.symbCache[key] = blank
        return blank


'''
    Functions to simplify expressions containing symbols
'''


def getExprSubstituted(expr, symbsAndValues, numeric=False):

    if not isinstance(expr, sp.Expr):
        return expr

    # attempt to substitute every symbol within expr with a value
    for symb in expr.free_symbols:
        if symb in symbsAndValues:
            value = symbsAndValues[symb]
            expr = expr.subs(symb, value)

    # if the expression has had all its symbols substituted, numerically evaluate it
    if numeric and len(expr.free_symbols) == 0:
        expr = expr.evalf()

    return expr


def getExprSimplified(expr, symbolic=True, numeric=False):

    if not isinstance(expr, sp.Expr):
        return expr

    # simplify by asserting symbols are real and positive
    # (if they were declared as such at creation)
    if symbolic:
        expr = expr.simplify()

    # assert 'probability' symbols are <1 via substitution,
    # as is needed to simplify e.g. sqrt expressions
    if symbolic:
        for symb in expr.free_symbols:
            assert isinstance(symb, Symb) or isinstance(symb, BlankSymb)

            if symb.is_probability:
                expr = expr.subs(symb, 1-symb).simplify().subs(symb, 1-symb)

    # if the expression contains no symbols, eval it as floating-point
    if numeric and len(expr.free_symbols) == 0:
        expr = expr.evalf()

    return expr



'''
    Functions to validate symbol substitutions satisfy assumptions
'''

def assertSymbValuesSatisfyAssumptions(symbsAndValues):

    # this is NOT a comprehension validation of all sympy Symbol assumptions;
    # we only validate that the assumptions of being real or a probability
    # are consistent with their given values (when not symbolic). We neglect
    # checks that (e.g.) values are integers when symb.is_integer, etc

    for symb, value in symbsAndValues.items():

        # all symbols must be Symb instances (or subclasses); NOT sp.Symbol
        # (because we need method .is_probability to exist)
        assert isinstance(symb, Symb), f'argument {symb} was not a Symb, Param or Prob'

        # assume substitution of other symbols/variables satisfies assumptions
        if isinstance(value, sp.Expr) and len(value.free_symbols) > 1:
            continue

        # replace constant expressions (e.g. 1+pi/2) with floating-point during check
        if isinstance(value, sp.Expr):
            value = value.evalf() # returns sp.Expr, or sp.Float, etc

        # replace zero-imaginary-component complex numbers with real floats during check
        if isinstance(value, complex) and value.imag == 0:
            value = value.real

        # we don't need an equivalent zero-imag replacement for numerical sympy expressions,
        # since evalf() above should have automatically removed zero imaginary components 
        
        # probabilities must be real numbers in [0,1]
        if symb.is_probability:
            try:
                # duck-type check that number is real
                assert 0 <= value <= 1, f'Probability {symb} was assigned value {value} which is outside [0,1]'
            except AssertionError:
                raise
            except:
                assert False, f'Probability {symb} was assigned a non-real value of {value} (of type {type(value).__name__})'

        # real numbers must be real (obviously)
        if symb.is_real:
            try:
                # which means their duck-typed comparison with any real number must succeed
                _ = value < len('Manuel Rudolph loves suffix tries')
            except:
                assert False, f'Real-valued symbol {symb} was assigned a non-real value of {value} (of type {type(value).__name__})'

        # TODO: here, you could validate further constraints/assumptions are satisfied
