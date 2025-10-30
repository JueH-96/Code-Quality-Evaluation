import sys

# Read input from stdin
N = int(sys.stdin.readline())
P = list(map(int, sys.stdin.readline().split()))

# The problem asks for the minimum cost to sort the permutation P_1, ..., P_N
# to 1, ..., N using adjacent swaps. The cost of swapping P_i and P_{i+1} is i.
# This problem can be transformed by considering the sequence a_i = i - P_i.
# The target state is P_i = i for all i, which means a_i = i - i = 0 for all i.
# An operation swapping P_j and P_{j+1} at cost j changes the sequence a.
# Let P'_j = P_{j+1} and P'_{j+1} = P_j.
# The new a'_j = j - P'_j = j - P_{j+1}.
# The new a'_{j+1} = (j+1) - P'_{j+1} = (j+1) - P_j.
# The original a_j = j - P_j and a_{j+1} = (j+1) - P_{j+1}.
# Notice that a'_j = (j+1 - P_{j+1}) - 1 = a_{j+1} - 1.
# And a'_{j+1} = (j - P_j) + 1 = a_j + 1.
# So the operation on P is equivalent to the operation on a:
# (a_j, a_{j+1}) -> (a_{j+1} - 1, a_j + 1) at cost j.
# We want to transform the initial sequence a to (0, 0, ..., 0) using these operations.

# Let S_k be the prefix sum of a: S_k = sum_{i=1}^k a_i.
# How does the operation at index j affect the prefix sums?
# S'_k = S_k for k < j.
# S'_j = S_{j-1} + a'_j = S_{j-1} + (a_{j+1} - 1).
# S'_{j+1} = S'_j + a'_{j+1} = (S_{j-1} + a_{j+1} - 1) + (a_j + 1) = S_{j-1} + a_j + a_{j+1} = S_{j+1}.
# S'_k = S_k for k > j+1.
# The operation at index j changes only S_j, by Delta S_j = (S_{j-1} + a_{j+1} - 1) - (S_{j-1} + a_j) = a_{j+1} - a_j - 1.

# This specific transformation on prefix sums (only S_j changes, by an amount depending on current state)
# with cost j is known to result in a minimum total cost equal to the sum of absolute values of the initial prefix sums of a.
# The target prefix sums are S_k = sum_{i=1}^k 0 = 0 for all k.

# Calculate a_i = i - P_i using 1-based indexing for i and P_i.
# P is 0-indexed in Python list P[0]...P[N-1].
# a_i corresponds to the value at index i in 1-based indexing.
# We will use a 1-based list for 'a'.
a = [0] * (N + 1)
for i in range(1, N + 1):
    a[i] = i - P[i-1] # P[i-1] is the value P_i in 1-based indexing

# Calculate prefix sums S_k = sum_{i=1}^k a_i for k=1..N.
# Using a 1-based list for 'S'. S[0] = 0.
S = [0] * (N + 1)
for k in range(1, N + 1):
    S[k] = S[k-1] + a[k]

# The minimum cost is sum of absolute values of prefix sums S_k for k=1 to N-1.
total_cost = 0
for k in range(1, N): # Sum over k from 1 to N-1
    total_cost += abs(S[k])

# Print the result to stdout
print(total_cost)