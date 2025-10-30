# YOUR CODE HERE
import sys
from itertools import permutations

def can_rearrange_strings(N, M, strings):
    def is_valid_sequence(seq):
        for i in range(len(seq) - 1):
            diff_count = sum(1 for a, b in zip(seq[i], seq[i + 1]) if a != b)
            if diff_count != 1:
                return False
        return True

    for perm in permutations(strings):
        if is_valid_sequence(perm):
            return "Yes"
    return "No"

# Read input
input = sys.stdin.read
data = input().split()
N = int(data[0])
M = int(data[1])
strings = data[2:]

# Solve the problem
result = can_rearrange_strings(N, M, strings)

# Print the result
print(result)