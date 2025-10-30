import sys
from itertools import combinations

# Read input
input = sys.stdin.read
data = input().split()

N = int(data[0])
T = int(data[1])
M = int(data[2])

incompatible_pairs = []
for i in range(M):
    A = int(data[3 + 2 * i]) - 1
    B = int(data[4 + 2 * i]) - 1
    incompatible_pairs.append((A, B))

# Function to check if a division is valid
def is_valid_division(division):
    for A, B in incompatible_pairs:
        if division[A] == division[B]:
            return False
    return True

# Generate all possible divisions
valid_divisions = 0
for division in combinations(range(T), N):
    division = list(division)
    for i in range(N):
        division[i] = division[i] % T
    if is_valid_division(division):
        valid_divisions += 1

# Print the result
print(valid_divisions)