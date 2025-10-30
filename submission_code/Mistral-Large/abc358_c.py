import sys
from itertools import combinations

def min_stands_to_visit(N, M, stands):
    # Convert the input strings to sets of flavors for easier manipulation
    flavor_sets = []
    for s in stands:
        flavor_set = {j for j in range(M) if s[j] == 'o'}
        flavor_sets.append(flavor_set)

    # Check all combinations of stands from 1 to N
    for r in range(1, N + 1):
        for combo in combinations(flavor_sets, r):
            # Union of flavors in the current combination
            combined_flavors = set().union(*combo)
            if len(combined_flavors) == M:
                return r

    return N  # In case all stands need to be visited (though constraints ensure this won't happen)

# Read input
input = sys.stdin.read
data = input().split()
N = int(data[0])
M = int(data[1])
stands = data[2:2+N]

# Calculate and print the result
result = min_stands_to_visit(N, M, stands)
print(result)