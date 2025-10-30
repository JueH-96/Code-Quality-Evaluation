# YOUR CODE HERE
import sys

# Read the number of elements, N
N = int(sys.stdin.readline())

# Read the N integers into a list A
# The input line contains space-separated integers
A = list(map(int, sys.stdin.readline().split()))

# Check if the list A is strictly increasing
# A sequence A is strictly increasing if A[i] < A[i+1] for all valid indices i.
# In 0-based indexing, this means A[i] must be strictly less than A[i+1] for i from 0 to N-2.
# The generator expression (A[i] < A[i+1] for i in range(N - 1)) creates a sequence of boolean values.
# range(N - 1) generates indices from 0 up to N-2.
# The all() function returns True if all elements in the generated sequence are True, and False otherwise.
# This effectively checks if A[0] < A[1], A[1] < A[2], ..., and A[N-2] < A[N-1] are all true.
is_strictly_increasing = all(A[i] < A[i+1] for i in range(N - 1))

# Print "Yes" if strictly increasing, otherwise print "No".
# The judge is case-insensitive, so "Yes" and "No" are acceptable.
if is_strictly_increasing:
    print("Yes")
else:
    print("No")