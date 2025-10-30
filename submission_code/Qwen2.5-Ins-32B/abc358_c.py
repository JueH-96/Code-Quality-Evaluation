import itertools

def min_stands_to_visit(N, M, stands):
    # Convert each stand's string to a set of flavors it sells
    stands_sets = [set() for _ in range(N)]
    for i in range(N):
        for j in range(M):
            if stands[i][j] == 'o':
                stands_sets[i].add(j)
    
    # Generate all combinations of stands
    for r in range(1, N + 1):
        for combination in itertools.combinations(stands_sets, r):
            # Check if the combination covers all flavors
            if set.union(*combination) == set(range(M)):
                return r
    return N  # In case no combination is found, which shouldn't happen given the constraints

# Read input
N, M = map(int, input().split())
stands = [input().strip() for _ in range(N)]

# Solve and print the result
print(min_stands_to_visit(N, M, stands))