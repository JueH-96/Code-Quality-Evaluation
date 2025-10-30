import sys

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # A sequence with N=2 terms is always considered a geometric progression.
    # The common ratio is A[1]/A[0].
    # Our loop condition range(1, N-1) naturally handles this:
    # If N=2, range(1, 1) is empty, so the loop doesn't run, and 'is_gp' remains True.

    # For a sequence to be a geometric progression, the ratio A_{i+1}/A_i must be constant
    # for all i from 1 to N-1.
    # We determine the reference common ratio from the first two terms (A[0] and A[1]).
    # To avoid floating-point precision issues and to correctly handle non-integer ratios,
    # we represent the common ratio as a fraction: common_ratio_num / common_ratio_den.
    # The first ratio is A[1] / A[0].
    common_ratio_num = A[1]
    common_ratio_den = A[0]

    is_gp = True
    # Iterate from the third term (index 2) up to the last term (index N-1).
    # For each A[i], we check if the ratio A[i]/A[i-1] matches the initial ratio A[1]/A[0].
    # In other words, for each i from 1 to N-2, we compare A[i+1]/A[i] with A[1]/A[0].
    # This comparison A[i+1]/A[i] == common_ratio_num / common_ratio_den
    # is mathematically equivalent to A[i+1] * common_ratio_den == A[i] * common_ratio_num
    # (assuming common_ratio_den and A[i] are non-zero, which they are given A_i >= 1).
    for i in range(1, N - 1):
        # Check if the ratio of the current pair (A[i], A[i+1]) matches the reference ratio
        if A[i+1] * common_ratio_den != A[i] * common_ratio_num:
            is_gp = False
            break # No need to check further if a mismatch is found

    if is_gp:
        print("Yes")
    else:
        print("No")

# Call the solve function to run the program
solve()