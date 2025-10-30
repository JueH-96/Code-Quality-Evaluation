import sys

# Read N
N = int(sys.stdin.readline())

# Read the sequence A
# Read the second line, split by whitespace, and convert each part to an integer
A = list(map(int, sys.stdin.readline().split()))

# Check if the sequence is strictly increasing
# A sequence A is strictly increasing if A[i] < A[i+1] for all i from 0 to N-2.
# We can iterate through pairs of consecutive elements using zip.
# A[:-1] creates a list from the first element up to the second-to-last.
# A[1:] creates a list from the second element up to the last.
# zip combines these into pairs (A[0], A[1]), (A[1], A[2]), ..., (A[N-2], A[N-1]).
# The generator expression x < y for x, y in zip(A[:-1], A[1:])
# checks if the first element is less than the second for each pair.
# all() returns True if the condition is true for all pairs, False otherwise.
if all(x < y for x, y in zip(A[:-1], A[1:])):
    # If all pairs satisfy the strictly increasing condition
    print("Yes")
else:
    # If any pair does not satisfy the condition
    print("No")