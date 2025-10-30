import math
from itertools import permutations

def find_sequence(N, K):
    # Generate the initial sequence
    initial_sequence = [i for i in range(1, N+1) for _ in range(K)]
    # Generate all permutations of the initial sequence
    all_permutations = sorted(set(permutations(initial_sequence)))
    # Calculate the index of the desired sequence
    index = math.floor((len(all_permutations) + 1) / 2) - 1
    # Return the desired sequence
    return all_permutations[index]

# Read input
N, K = map(int, input().split())

# Find and print the desired sequence
result = find_sequence(N, K)
print(' '.join(map(str, result)))