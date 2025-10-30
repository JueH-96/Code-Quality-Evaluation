# YOUR CODE HERE
import sys

def solve():
    # Read the length of the sequence
    n = int(sys.stdin.readline())
    # Read the sequence elements into a list of integers
    # Assumes the input format is correct as specified (N integers separated by spaces)
    a = list(map(int, sys.stdin.readline().split()))

    # A sequence of length 2 is always considered a geometric progression
    # because there's only one ratio to consider (or no ratio comparison needed).
    # The loop below handles this correctly as range(2, 2) is empty.

    # Assume the sequence is a geometric progression initially
    is_geometric = True

    # Check the geometric progression condition using cross-multiplication.
    # A sequence A is geometric if the ratio between consecutive terms is constant.
    # That is, A[k] / A[k-1] = constant for k = 1, 2, ..., N-1.
    # This implies A[k] / A[k-1] == A[k-1] / A[k-2] for k = 2, 3, ..., N-1.
    # To avoid floating-point inaccuracies, we use cross-multiplication:
    # A[k] * A[k-2] == A[k-1] * A[k-1]
    # This check needs to be performed for k ranging from 2 up to n-1.
    # The loop iterates from k=2 (using indices 0, 1, 2) up to k=n-1 (using indices n-3, n-2, n-1).
    for k in range(2, n):
        # Constraints state A_i >= 1, so all elements are positive integers.
        # This means a[k-1] and a[k-2] are guaranteed to be non-zero.
        # Python's integers support arbitrary precision, handling large products (up to 10^9 * 10^9 = 10^18).
        if a[k] * a[k-2] != a[k-1] * a[k-1]:
            # If the cross-multiplication equality does not hold for any k,
            # the ratios are not constant, and it's not a geometric progression.
            is_geometric = False
            # We can stop checking as soon as one mismatch is found.
            break

    # Output the final result based on whether the condition held for all checks.
    if is_geometric:
        print("Yes")
    else:
        print("No")

# Execute the solve function to run the program logic
solve()