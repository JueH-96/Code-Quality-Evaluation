# YOUR CODE HERE
from itertools import combinations

def min_stands_to_visit(N, M, stands):
    all_flavors = set(range(M))
    
    # Convert stands to list of sets of flavors
    stands_flavors = []
    for stand in stands:
        flavors = {i for i, ch in enumerate(stand) if ch == 'o'}
        stands_flavors.append(flavors)
    
    # Try combinations of stands to find the minimum number of stands needed
    for r in range(1, N + 1):
        for combo in combinations(stands_flavors, r):
            combined_flavors = set()
            for flavors in combo:
                combined_flavors.update(flavors)
            if combined_flavors == all_flavors:
                return r

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
stands = data[2:]

# Get the result
result = min_stands_to_visit(N, M, stands)

# Print the result
print(result)