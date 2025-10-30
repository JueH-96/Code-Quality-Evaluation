# YOUR CODE HERE
from itertools import permutations
from math import floor

def find_good_sequence(N, K):
    # Generate the base sequence
    base_sequence = []
    for i in range(1, N + 1):
        base_sequence.extend([i] * K)
    
    # Generate all unique permutations of the base sequence
    all_permutations = set(permutations(base_sequence))
    
    # Sort the permutations lexicographically
    sorted_permutations = sorted(all_permutations)
    
    # Find the floor((S+1)/2)-th permutation
    target_index = floor((len(sorted_permutations) + 1) / 2) - 1
    target_sequence = sorted_permutations[target_index]
    
    # Convert the sequence to a space-separated string
    result = ' '.join(map(str, target_sequence))
    
    return result

# Read input
N, K = map(int, input().split())

# Find and print the desired sequence
print(find_good_sequence(N, K))