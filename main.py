
from sympy import sqrt

from source . twiddles import getStringOfPauliInd
from source . symbols import Symb, Param, Prob
from source . ensemble import WeightedSumOfPauliStrings
from source . operators import H, Rx, Ry, Rz, U, Damp, Deph, Depol, Kraus, PauliGadget



# prepare our initial Pauli string
state = WeightedSumOfPauliStrings()
state.addPauliString('X', 1)
state.addPauliString('YIZ', -2)

# which can even include symbols
a = Symb('a')
state.addPauliString('ZZZ', a)

# and expressions thereof:
b = Symb('b')
state.addPauliString('ZXY', sqrt(a)+b**2)



# prepare a circuit
circ = [H(0), H(1), H(2)]

# which can include parameterised gates
c = Param('c')
d = Param('d')
circ += [Rx(0, c), Ry(1, d)]

# and control-qubits
e = Param('e')
circ += [Rz(2, e, ctrls=[0]), H(0, ctrls=[1,2])]

# and Pauli gadgets
f = Param('f')
circ += [PauliGadget([1,2], 'XY', f)]

# and error channels
g = Prob('g')
h = Prob('h')
circ += [Damp(0, g), Depol([0,1], h)]



# let's apply the circuit so far...
for op in circ:
    print('applying', op)
    state.applyOperator(op)

# and take a peek at the symbolic weights
print('\nYYZ =', state.getWeight('YYZ'), '\n')

# getWeight() triggered a heirarchal symbolic evaluation,
# which we now clear in order to apply additional gates
state.clearWeights()
num = len(circ)



# we can also describe general complex multi-param matrices
j = Symb('j')
k = Symb('k')
matr = (
  (0, 0, 1j, 0),
  (0, j+k, 0, 0),
  (j, 0, 0, j**2),
  (0, 0, 0, 0))
circ += [U([1,2], matr)]
circ += [U([0,2], matr, ctrls=[1])]

# and channels as Kraus maps
kraus1 = ((0, j), (k, 0))
kraus2 = ((j+k, 0), (k**2, sqrt(j)))
krauses = (kraus1, kraus2)
circ += [Kraus(0, krauses)]
circ += [Kraus([1,2], (matr,))]

# let's apply these remaining operators
for op in circ[num:]:
    print('applying', op)
    state.applyOperator(op)

print('')



# our operator PTMs are now cached, so re-applying them is much faster:
num = len(circ)
circ *= 3
for op in circ[num:]:
    print('faster applying', op)
    state.applyOperator(op)

print('')

# even if we change the parameters and qubits:
num = len(circ)
circ += [
    Rz(2, c, ctrls=[0]),
    Rz(1, d, ctrls=[2]), 
    Rz(0, e, ctrls=[1]),
    Rz(0, f, ctrls=[2]), 
    H(2, ctrls=[0,1]),
    H(1, ctrls=[2,0]),
    Kraus(0, krauses),
    Kraus(1, krauses),
    Kraus(2, krauses),
    U([0,1], matr, ctrls=[2]),
    U([2,0], matr, ctrls=[1]),
    U([2,1], matr, ctrls=[0])
]

for op in circ[num:]:
    print('still faster applying', op)
    state.applyOperator(op)

print('')



# we can even use Symb() in-place of Param() and Prob() as operator parameters,
# permitting us to later substitute in complex or unbounded scalars, violating CPTP

l = Symb('l')
g1 = Ry(0, c) # real
g2 = Ry(1, l) # complex

for op in [g1, g2]:
    print(op, ':\n', op.getSuperOperator(), '\n')

m = Symb('m')
g3 = Damp(0, g) # probability
g4 = Damp(1, m) # complex

for op in [g3, g4]:
    print(op, ':\n', op.getSuperOperator(), '\n')

# let's apply these non_CPTP operators
circ += [g2, g4]
for op in circ[-2:]:
    print('applying non-CPTP', op)
    state.applyOperator(op)

print('')



# symbolically evaluating a Pauli string weight would now be
# infeasibly expensive, so we instead set numerical values:
values = {
    a: 1+1j, b: -1-0.5j,     # Symb
    c: 1, d: 2, e: 3, f: 4,  # Param
    g: .1, h:.1,             # Prob
    j: 1j, k: 2j, l: 1, m:2  # Symb
}

# binding them to the state accelerates subsequent weight calculation
state.setParams(values)

numQubits = state.getNumQubits()
numStates = state.getNumPossibleStrings()

for ind in range(numStates):
    weight = state.getWeight(ind)
    print(getStringOfPauliInd(ind, numQubits), '=', weight)

print('')



# we do not have to assign all symbols a value; we can keep some symbolic
n = Param('n')
p = Prob('p')
q = Symb('q')

num = len(circ)
circ += [
    PauliGadget([0,1,2], 'XYZ', n),
    Deph([0,1], p),
    Ry([2], q, ctrls=[1])
]

# apply operators parameterised by (n, p, q)
state.clearWeights()

for op in circ[num:]:
    print('applying', op)
    state.applyOperator(op)

# substitute values for all symbols except n,p,q
state.setParams(values)

# obtain a weight in terms of only n, p, q
print('\nXYZ =', state.getWeight('XYZ'))

# give remaining symbols values
values.update({n: -.1, p: .2, q: -1-1j})
state.setParams(values)



# compute the main output; the overlap with |0>
print('\ncalculating overlap...')
overlap = state.getZeroOverlap()
print('\noverlap with |0> =', overlap)



# we can accelerate simulation via truncation
print('\nclearing simulation\n')
state.clearOperators()

for i, op in enumerate(circ):
    print('applying', op)
    state.applyOperator(op)

    # every 4 operators, let's kill strings of <1% lineage
    if not (i+1)%4:
        print('\ttruncating...')
        state.truncateByFracLineage(.01)

print('\ncalculating overlap...')
trunced = state.getZeroOverlap()
print('\ntruncated overlap with |0> =', trunced)

# this is likely catastrophic for only 3 qubits, and our
# ridiculously non-CPTP operators...
print('\nerror =', abs((overlap-trunced)/overlap)*100, '%\n')