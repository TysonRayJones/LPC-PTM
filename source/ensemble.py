'''
    This file defines the data structures and logic to maintain a
    (symbolically) weighted sum of Pauli strings, and evolve them
    under the action of Pauli transfer matrices obtained at runtime
    from canonical or general operators. Evolution is performed 
    "breadth-first", so that the ensemble of stored Pauli strings
    (generally) grows with circuit depth.

    There are two main components of this file's design:

      - class 'WeightedSumOfPauliStrings' which a user instantiates,
        adds (symbolically weighted) initial Pauli strings to, and
        applies the operators (from operators.py) to. This class
        contains the logic for modifying a Pauli string under a PTM
        in an operator-agnostic way, or through optimised logic
        specific to certain operators (e.g. PauliGadgets). It also
        contains example logic to "truncate" the ensemble, as used 
        by the LOWESA algorithm: https://arxiv.org/abs/2308.09109,
        and logic to efficiently "merge" Pauli strings output by an 
        operator into the existing ensemble.

      - class '_WeightEvalGraph' which is a graph populated by
        WeightedSumOfPauliStrings during simulation, encoding
        the simulation history necessary to calculate the final
        weights of the output Pauli string. The history is kept
        symbolic, but stored in an efficient heirarchal manner; it
        costs O(1) time to log during simulation, and O(#edges) to
        obtain all output Pauli string weights (by performing a
        heirarchal reductoin). 

    The main innovations of this file (and full repository) are:

      - support of completely general operators and channels, 
        described by Z-basis matrices or Kraus maps, containing
        any number of user-given parameters and symbols. Bespoke
        optimised treatments of specific gates are also supported;
        we demonstrate herein Manuel Rudolph's handling of Pauli
        gadgets using the generator commutator.

      - runtime caching of calculated Pauli transfer maps of said
        operators.

      - support of general parameter values which do not necessarily
        yield completely-positive trace-preserving channels; for
        instance, complex Rx gate strengths, or negative error
        probabilities.

      - handling of symbolic parameters (and amplitudes) as first-
        class citizens, using Sympy, whilst deferring their 
        expensive simplification and/or substitution to posteriori
        expectation-value evaluation. This permits users to
        post-simulation substitute some symbols while retaining
        others when evaluating quantities like Pauli string overlaps.

      - storage of constituent Pauli strings in a dictionary, rather
        than an unsorted list type.  This enables efficient...

      - runtime merging of incident Pauli strings produced by the
        same operator upon differnet input Pauli strings.

      - a very beautiful, readable, modular, extensible, and
        defensively-designed software architecture ;)
        In particular, the modularity of WeightedSumOfPauliStrings
        (i.e. its loose-coupling from the logics of operators.py
        and paulitransfer.py) allows us to later substitute it
        with a class using a different internal representation
        of a Pauli ensemble, to the current use of a dictionary,
        to accelerate simulation. For example, suffix tries:
        https://arxiv.org/abs/2403.11644

    The design of this repo precludes:

      - "depth-first" simulation to bound runtime memory costs
      - "fixed-angle" simulation to improve truncations; this is
        straightforward to implement and optionally deploy, but 
        would complicate the otherwise simple code.
'''


from . twiddles import (
    getBit,
    setBit,
    getSubIndFromPauliStringInd, 
    setSubIndOfPauliStringInd, 
    getOperatorWeightOfPauliStringInd,
    getNumPaulisInStringInd,
    getIndOfPauliString,
    getIndOfPauliStringProduct,
    doPauliStringIndsCommute)

from . symbols import (
    getExprSimplified, 
    getExprSubstituted,
    assertSymbValuesSatisfyAssumptions)

from . operators import PauliGadget



class WeightedSumOfPauliStrings():
    '''
        Dictionary-based weighted sum of Pauli strings,
        where Pauli strings are represented as infinite-
        precision integers (implicitly unsigned)
        interpreted in base-4.
    '''

    def __init__(self):

        # a dict of Pauli-string indexes (big integers) to their corresponding WeightEvalGraph node.
        # this dict only ever contains the leaf layer of the graph, with pointers upward to ancestors
        self.strings = {}

        # same as .strings but stores only the root nodes (with no pointers); needed only by clearOperators()
        self.roots = {}

        # a workspace for copying strings for safe key updating
        self.workspace = {}

        # a dict of operator param symbols (potentially a subset of operators) to numerical values,
        # which is user supplied for posteriori weight-calculation
        self.paramValues = {}

        # a lock to prevent adding Pauli strings to ensemble after maps have been applied 
        self.disableAddPauliString = False

        # a lock to prevent applying operators after parameter substitution has begun
        self.disableApplyOperator = False


    def getNumQubits(self):
        return getNumPaulisInStringInd(max(self.strings))
    

    def getNumStrings(self):
        return len(self.strings)
    

    def getNumPossibleStrings(self):
        return 4 ** self.getNumQubits()


    def addPauliString(self, ind, weight):

        # must not add strings mid-simulation because node tree depth would then be
        # inconsistent and break the posteriori weight calculation. Specifically,
        # only root nodes may have an initially non-None weight.
        assert not self.disableAddPauliString

        # convenience overload to support ind given as 'IXYZ' string
        if isinstance(ind, str):
            return self.addPauliString(getIndOfPauliString(ind), weight)

        # create a new node for this string, if it doesn't already exist
        # (below Node may get instantly garbage collected; I don't care)
        self.strings.setdefault(ind, _WeightEvalGraph.Node(isRoot=True, weight=0))

        # add to the string's weight
        self.strings[ind].weight += weight

        # In theory, if the weight became 0, we could remove the string here, saving all
        # its wasteful later processing. But this is an unlikely scenario; why
        # would the user be giving self-cancelling initial strings? Best to do nothing.

        # renormalise these root-leaf truncation attributes
        frac = 1./len(self.strings)
        for node in self.strings.values():
            node.fractionOfLineage = frac

        # lazily shallow-copy strings into roots, because disableAddPauliString precludes non-roots in strings
        self.roots = self.strings.copy()


    def applyOperator(self, operator):

        # must not apply further operators after parameter substitution has
        # begun. This is because the tree already contains nodes with non-None
        # weights, which prevents further upward recursion when queried. So
        # applying another operator would add a new leaf layer with None
        # weights, occluding the non-None ancestors, causing early termination
        # when clearing or evaluating weights (which terminate when encountering
        # a non-None). One sympton would be that new parameter substitutions 
        # are not heeded when obtaining weights; a bug/slow problem!
        assert not self.disableApplyOperator

        # disable subsequent calls to addPauliString()
        self.disableAddPauliString = True

        # empty workspace, which we will subsequently re-populate
        self.workspace.clear()

        # use bespoke method, or treatment of operator as a generic map
        if isinstance(operator, PauliGadget):
            self._applyPauliGadget(operator)
        else:
            self._applyOperatorViaMap(operator)
    
        # overwrite strings with workspace, in effect adjusting .strings to the next layer
        self.strings.clear()
        self.strings.update(self.workspace)
   

    def _applyOperatorViaMap(self, operator):

        # get PTM of the operator, creating or using a cached version
        map = operator.getPauliTransferMap()
        qubits = operator.getQubits()

        # consider applying map to each string in the ensemble, in-turn.
        # it's important to keep this large loop parallelisable, 
        # and ergo that we carefully leverage dict atomicity, etc.
        for initInd in self.strings:

            # consider the map modifying only the targeted sub-string
            initSubInd = getSubIndFromPauliStringInd(initInd, qubits)
            outSubInds = map[initSubInd]

            # integrate the resulting sub-strings into the workspace
            self._mergeSubstatesIntoWorkspace(operator, initInd, initSubInd, outSubInds)
 

    def _applyPauliGadget(self, gadget):

        # get the Pauli-string generator of the gadget
        genInd = gadget.getGeneratorInd()
        qubits = gadget.getQubits()

        # consider applying gadget to each string in the ensemble, in-turn
        for initInd in self.strings:

            # consider the gadget modifying only the targeted sub-string
            initSubInd = getSubIndFromPauliStringInd(initInd, qubits)

            # obtain the 1 or 2 sub-strings produced by a gadget upon initSubInd,
            # which always includes the input substring. Note that the below logic
            # (i.e. that commuting generator & substring imply no-branching) only
            # holds when the gadget's symbolic parameter is strictly real. In the
            # general case (the parameter can be complex), the gadget always invokes
            # two-branching. For simplicity, we do not demonstrate this latter case.

            outSubInds = [initSubInd]
            if not doPauliStringIndsCommute(initSubInd, genInd):
                prodInd = getIndOfPauliStringProduct(initSubInd, genInd)
                outSubInds.append(prodInd)

            # integrate the resulting sub-strings into the workspace
            self._mergeSubstatesIntoWorkspace(gadget, initInd, initSubInd, outSubInds)

                
    def _mergeSubstatesIntoWorkspace(self, operator, initInd, initSubInd, outSubInds):

        # obtain the evaluation-tree node bound to the input Pauli string
        initNode = self.strings[initInd]

        # prepare truncation measures (beware; custom gates permit empty outSubInds)
        outLineageFac = initNode.fractionOfLineage / max(len(outSubInds), 1)

        # process each output Pauli sub-string
        for i, outSubInd in enumerate(outSubInds):

            # obtain the resulting full string
            outInd = setSubIndOfPauliStringInd(initInd, outSubInd, operator.getQubits())

            # PauliGadgets don't record out state ind; they record branch-or-not flag,
            # and we always receive outSubInds in order: [initSubInd, (optional) branchInd]
            if isinstance(operator, PauliGadget):
                outSubInd = i

            # bundle the info necessary to link the new string to its parent node (i.e. initNode)
            edgeInfo = (initNode, operator, initSubInd, outSubInd)

            # incorporate the new string
            self._mergePauliStringIntoWorkspace(outInd, edgeInfo, outLineageFac)


    def _mergePauliStringIntoWorkspace(self, stringInd, edgeInfo, outLineageFac):
        
        # using existing node for string, else create a new one
        if stringInd in self.workspace:
            node = self.workspace[stringInd]
        else:
            node = _WeightEvalGraph.Node()
            self.workspace[stringInd] = node
        
        # link node to parent with necessary relative info for later determining child weight
        edge = _WeightEvalGraph.Edge(*edgeInfo)
        node.edges.append(edge)

        # update the node's truncation attributes
        node.fractionOfLineage += outLineageFac


    def truncateByFracLineage(self, minFrac):

        if self.disableApplyOperator:
            raise RuntimeError('Cannot truncate initial Pauli strings')

        # remove all leaves with smaller lineage frac than that given
        for ind, node in list(self.strings.items()): # immutable iterator
            if node.fractionOfLineage < minFrac:

                # because nodes point upward, garbage collector kills all unique ancestors
                del self.strings[ind]

        self._normaliseTruncationMetrics()

    
    def truncateByOperatorWeight(self, maxWeight):

        if self.disableApplyOperator:
            raise RuntimeError('Cannot truncate initial Pauli strings')

        # determine the operator weight (number of non-identities) of each leaf node
        for ind in list(self.strings.keys()):
            weight = getOperatorWeightOfPauliStringInd(ind)

            # kill leaf nodes with exceed the max weight
            if weight > maxWeight:

                # and because nodes point upward, garbage collector kills all unique ancestors
                del self.strings[ind]

        self._normaliseTruncationMetrics()


    def _normaliseTruncationMetrics(self):

        if not self.strings:
            raise RuntimeError('Removed all Pauli strings during truncation')
        
        total = sum(node.fractionOfLineage for node in self.strings.values())
        for node in self.strings.values():
            node.fractionOfLineage /= total


    def setParams(self, paramValues):
        assertSymbValuesSatisfyAssumptions(paramValues)

        # changing param values invalidates existing weight calculations
        self.clearWeights()

        # record the param values, which get dynamically substituted during getWeight()
        self.paramValues = paramValues


    def getWeight(self, ind, simplify=False, numeric=True):

        # convenience overload to support ind given as 'IXYZ' string
        if isinstance(ind, str):
            return self.getWeight(getIndOfPauliString(ind), simplify, numeric)

        # disable subsequent calls to applyOperator()
        self.disableApplyOperator = True

        # any string not present has a weight of zero
        if ind not in self.strings:
            return 0
        
        # compute string's weight which may invoke tree evaluation
        leaf = self.strings[ind]
        weight = leaf.getWeight(self.paramValues)

        # weight may retain symbols (to be simplified) or might not (convert to floating-point)
        weight = getExprSimplified(weight, simplify, numeric)
        return weight
    

    def clearWeights(self):
        
        # clear weight of all leaf nodes, which will automatically clear ancestor weights
        for node in self.strings.values():
            node.clearWeights()

        # this permits subsequent application of more operators
        self.disableApplyOperator = False


    def getZeroOverlap(self):
        num = self.getNumQubits()
        total = 0

        # iterate all Pauli states of |0> = (tensor) (I+Z)
        for i in range(2 ** num):

            # get ind of IZZI...
            ind = 0
            for j in range(num):
                b = getBit(i, j)
                ind = setBit(ind, 2*j,   b)
                ind = setBit(ind, 2*j+1, b)

            # compute and combine its weight
            total += self.getWeight(ind)

        return total
    

    def clearOperators(self):

        # erase evaluation history by setting roots as leafs (all nodes/edges are gc'd)
        self.strings = self.roots.copy()

        # enable adding Paulis and applying operators
        self.disableAddPauliString = False
        self.disableApplyOperator = False



class _WeightEvalGraph:
    '''
        A graph which encodes the evaluation history
        of a PTM simulation. Nodes correspond to Pauli
        strings and Edges correspond to operators which
        act upon the parent string to produce the child.
        The graph is used to posteriori evaluate the
        the coefficients of output Pauli strings, 
        symbolically or with numerical values substituted.

        The graph is 'upside down' in that nodes point
        upward to their parent and graph traversal begins
        at the leaf nodes. There are potentially multiple 
        root nodes (the initial strings) which are the 
        "deepest" to reach. 
    '''


    class Edge:

        def __init__(self, parent, operator, i, j):

            # edge points upward toward parent node
            self.parent = parent

            # the operator whose map upon the parent created this edge
            self.operator = operator

            # the indices within the operator's map which together identify a coefficient
            self.mapInInd = i
            self.mapOutInd = j


    class Node:

        def __init__(self, isRoot=False, weight=None, lineageFrac=0):

            # There may be multiple roots, corresponding to initial Pauli strings
            self.isRoot = isRoot

            # edges to parent nodes which created this child
            # node via a map. Root node(s) have an empty list
            self.edges = []

            # the coefficient of the Node's corresponding string, 
            # which is only non-None for the root node(s), or 
            # during post-simulation global weight evaluation.
            self.weight = weight

            # attributes for truncation
            self.fractionOfLineage = lineageFrac


        def getWeight(self, paramValues):

            # root node's weight is never modified, but it may be symbolic; substitute and return
            if self.isRoot:
                assert self.weight != None
                return getExprSubstituted(self.weight, paramValues, numeric=True)

            # return if weight was already computed (like if triggered by sibling)
            if self.weight is not None:
                return self.weight
            
            # otherwise compute weight from parents
            weight = 0
            for edge in self.edges:

                # compute the parents' weight (may trigger leaf-to-root traversal)
                parentWeight = edge.parent.getWeight(paramValues)

                # compute this operator's map's coefficient for this edge (replacing symbols with values or params).
                # this may potentially be a constant-free but non-numerical Sympy expression
                coeff = edge.operator.getMapCoeff(edge.mapInInd, edge.mapOutInd, paramValues)

                # add this parent's contribution to the child's weight
                weight += coeff * parentWeight

            # if no symbols are left, force evaluate the sympy expression to a numerical value
            weight = getExprSimplified(weight, symbolic=False, numeric=True)

            # bind weight so it is never re-computed for this node (nor ancestors)
            self.weight = weight
            return weight


        def clearWeights(self):

            # stop if this node (and its ancestors) are already cleared
            if self.weight is None:
                return
            
            # stop if this node is a root (do not clear it)
            if self.isRoot:
                return
            
            # otherwise, clear node and all ancestors
            self.weight = None
            for edge in self.edges:
                edge.parent.clearWeights()