import sys
import itertools
import functools
import operator

# Read input from stdin
data = sys.stdin.read().split()
index = 0
N = int(data[index])
K = int(data[index + 1])
index += 2
A = [int(data[index + i]) for i in range(N)]

# Compute the maximum XOR of any K distinct elements
max_xor = max(functools.reduce(operator.xor, combo) for combo in itertools.combinations(A, K))

# Output the result
print(max_xor)