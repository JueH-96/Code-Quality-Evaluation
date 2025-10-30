import sys
import math
from itertools import permutations

def calculate_value(initial_value, functions_sequence):
    """
    Calculates the value of applying a sequence of functions.
    functions_sequence is a list of (A, B, original_index) tuples,
    applied from first to last in the sequence.
    """
    current_value = initial_value
    for A, B, _ in functions_sequence:
        # Use Python's arbitrary precision integers
        current_value = A * current_value + B
    return current_value

# Read input
N, K = map(int, sys.stdin.readline().split())
functions = []
for i in range(N):
    A, B = map(int, sys.stdin.readline().split())
    functions.append((A, B, i)) # Store with 0-based index

# Sort functions by A descending to get candidates
functions_sorted_a_desc = sorted(functions, key=lambda x: x[0], reverse=True)

# Take the top M functions as candidates for the K positions
# The number of permutations P(M, K) should be feasible. K <= 10.
# P(12, 10) = 12! / (12-10)! = 12! / 2! = 239,500,800
# P(13, 10) = 13! / (13-10)! = 13! / 3! = 1,037,836,800
# M=12 seems safer for time limits. M=13 might also pass.
# M=min(N, 13) could be slightly more accurate if N is small, but constraints say N >= K >= 1.
# If N is smaller than 13, we just use all N functions anyway.
M = min(N, 13)
candidate_functions = functions_sorted_a_desc[:M]

max_value = 0

# Iterate over all permutations of size K from the candidate functions
# Each permutation p_sequence_tuple represents the tuple (p_1, p_2, ..., p_K)
# where f_{p_1}(f_{p_2}(... f_{p_K}(1) ... )) is the composition.
# The indices p_i refer to the original indices of the functions.
for p_sequence_tuple in permutations(candidate_functions, K):
    # The functions are applied in the order f_{p_K}, f_{p_{K-1}}, ..., f_{p_1}.
    # The sequence of functions to apply, from first applied to last applied,
    # is the reverse of the p_sequence tuple (p_1, ..., p_K).
    p_sequence = list(p_sequence_tuple)
    applied_sequence = p_sequence[::-1]

    # Calculate the value for this sequence starting with initial_value = 1
    current_value = calculate_value(1, applied_sequence)

    # Update max value
    if current_value > max_value:
        max_value = current_value

# Output the result
print(max_value)