# Define sets for sides and diagonals using frozensets
sides = {
    frozenset({'A', 'B'}),
    frozenset({'B', 'C'}),
    frozenset({'C', 'D'}),
    frozenset({'D', 'E'}),
    frozenset({'E', 'A'})
}

diagonals = {
    frozenset({'A', 'C'}),
    frozenset({'B', 'D'}),
    frozenset({'C', 'E'}),
    frozenset({'D', 'A'}),
    frozenset({'E', 'B'})
}

# Read input pairs
try:
    s1s2 = input().strip()
    t1t2 = input().strip()
except EOFError:
    s1s2 = ""
    t1t2 = ""

# Create frozensets for input pairs
pair1 = frozenset({s1s2[0], s1s2[1]})
pair2 = frozenset({t1t2[0], t1t2[1]})

# Determine if both pairs are either sides or diagonals
if (pair1 in sides and pair2 in sides) or (pair1 in diagonals and pair2 in diagonals):
    print("Yes")
else:
    print("No")