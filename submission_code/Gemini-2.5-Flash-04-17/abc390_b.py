# YOUR CODE HERE
import sys

# Read the number of terms
N = int(sys.stdin.readline())

# Read the sequence of positive integers
# A is a list A[0], A[1], ..., A[N-1]
A = list(map(int, sys.stdin.readline().split()))

# According to the constraints, 2 <= N <= 100.
# A sequence with N=2 terms (A_1, A_2) is always a geometric progression
# with first term A_1 and common ratio A_2 / A_1.
if N == 2:
    print("Yes")
else:
    # For N > 2, a sequence is a geometric progression if the ratio
    # between consecutive terms is constant.
    # The ratio of the first two terms is A[1] / A[0].
    # We need to check if the ratio A[k] / A[k-1] is equal to A[1] / A[0]
    # for all k from 2 to N-1.
    # Using cross-multiplication to avoid floating point issues:
    # A[k] * A[0] == A[k-1] * A[1]

    is_geometric = True
    # Iterate from the third term (index 2) up to the last term (index N-1).
    # The loop index k represents the 0-based index of the current term A[k].
    # We check if the ratio A[k]/A[k-1] matches the ratio A[1]/A[0].
    for k in range(2, N):
        # Check using cross-multiplication: A[k] * A[0] == A[k-1] * A[1]
        # Since all A_i >= 1, A[0] and A[k-1] are non-zero, division is safe if we were to use it,
        # but cross-multiplication is preferred for exact integer comparison.
        # The products A[k] * A[0] and A[k-1] * A[1] can be up to 10^9 * 10^9 = 10^18,
        # which fits within Python's arbitrary-precision integers.
        if A[k] * A[0] != A[k-1] * A[1]:
            is_geometric = False
            break # Found a non-constant ratio, it's not a GP

    # Print the result
    if is_geometric:
        print("Yes")
    else:
        print("No")