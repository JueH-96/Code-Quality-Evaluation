from itertools import permutations
from math import factorial

# Read input
grid = []
for _ in range(3):
    row = list(map(int, input().split()))
    grid.append(row)

# Generate all possible permutations of the grid
all_permutations = set(permutations([num for row in grid for num in row]))

# Function to check if a permutation causes disappointment
def causes_disappointment(perm):
    for i in range(4):
        if perm[i*3] == perm[i*3+1] and perm[i*3+2] != perm[i*3]:
            return True
    if perm[0] == perm[4] and perm[8] != perm[0]:
        return True
    if perm[2] == perm[4] and perm[6] != perm[2]:
        return True
    return False

# Count valid permutations
valid_permutations = 0
for perm in all_permutations:
    if not causes_disappointment(perm):
        valid_permutations += 1

# Calculate the probability
total_permutations = factorial(9)
probability = valid_permutations / total_permutations

print(probability)