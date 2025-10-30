# YOUR CODE HERE
import sys

# Read N
N = int(sys.stdin.readline())

# Read the sequence A
# Using 0-based indexing for list A, corresponding to A_1...A_N in problem
# A[0] corresponds to A_1, A[1] to A_2, ..., A[N-1] to A_N
A = list(map(int, sys.stdin.readline().split()))

# A sequence of length 2 (N=2) is always a geometric progression
# (since A_1 and A_2 are positive, the ratio A_2/A_1 is well-defined and positive).
# Our check handles N=2 correctly because the loop range(2, N) will be empty.

is_gp = True
# A sequence A is a geometric progression if A_k / A_{k-1} is constant for k=2, ..., N.
# This constant ratio must be equal to A_2 / A_1.
# So, we check if A_k / A_{k-1} == A_2 / A_1 for k = 3, ..., N.
# Using 0-based indexing (A[0], A[1], ...):
# Check if A[i] / A[i-1] == A[1] / A[0] for i = 2, ..., N-1.
# We use cross-multiplication to avoid floating-point issues:
# Check if A[i] * A[0] == A[i-1] * A[1] for i = 2, ..., N-1.
# The loop iterates through the 0-based indices i from 2 up to N-1.
for i in range(2, N):
    # Check if the ratio A[i]/A[i-1] is the same as A[1]/A[0]
    # which is equivalent to A[i] * A[0] == A[i-1] * A[1]
    # Since all A_k are positive integers (>= 1), A[i-1] and A[0] are >= 1,
    # so there's no division by zero issue if we were to use division.
    # The cross-multiplication approach using integers is safe and accurate.
    # A[i] * A[0] can be up to 10^9 * 10^9 = 10^18. Python handles this with arbitrary precision integers.
    if A[i] * A[0] != A[i-1] * A[1]:
        is_gp = False
        break

# Print the result
if is_gp:
    print("Yes")
else:
    print("No")