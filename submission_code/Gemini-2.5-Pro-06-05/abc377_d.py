# YOUR CODE HERE
import sys

def solve():
    """
    Reads input, solves the problem, and prints the answer.
    """
    # Use fast I/O
    input = sys.stdin.readline

    try:
        # Read N and M from the first line of input.
        line = input()
        if not line.strip():
            return
        N, M = map(int, line.split())
    except (IOError, ValueError):
        # Handles cases with empty or malformed input lines.
        return

    # A pair (l, r) is invalid if there exists an i such that l <= L_i and R_i <= r.
    # We will count the number of valid pairs directly.
    # A pair (l, r) is valid if for all i, it's NOT (l <= L_i and R_i <= r),
    # which is equivalent to (l > L_i or r < R_i).

    # We iterate through all possible l from 1 to M and for each l, count the
    # number of valid r's.
    # For a fixed l, r must satisfy:
    # 1. l <= r <= M
    # 2. For every i such that l <= L_i, we must have r < R_i.
    # This second condition means r must be smaller than the minimum of all R_i
    # for which L_i >= l.
    # Let f(l) = min({R_i | L_i >= l}). Then we need r < f(l).
    # So for a fixed l, the valid range for r is [l, min(M, f(l) - 1)].

    # To compute f(l) efficiently, we use two pre-computation steps.

    # Step 1: Create a helper array `min_R_at_L` where `min_R_at_L[c]` is the
    # minimum R_i among all intervals that start at L_i = c.
    # We use M+2 as a sentinel value, as it's larger than any possible R_i.
    sentinel = M + 2
    min_R_at_L = [sentinel] * (M + 2)
    for _ in range(N):
        L, R = map(int, input().split())
        min_R_at_L[L] = min(min_R_at_L[L], R)

    # Step 2: Compute f(l) = min({R_i | L_i >= l}) using dynamic programming.
    # The recurrence is f(l) = min(f(l+1), min_R_at_L[l]). We compute it backwards.
    f = [0] * (M + 2)
    f[M + 1] = sentinel
    for l in range(M, 0, -1):
        f[l] = min(f[l + 1], min_R_at_L[l])

    # Step 3: Iterate through l from 1 to M and sum up the counts of valid r's.
    total_valid_pairs = 0
    for l in range(1, M + 1):
        # Upper bound for a valid r is min(M, f[l] - 1).
        upper_bound_r = min(M, f[l] - 1)
        
        # Lower bound for r is l itself.
        lower_bound_r = l
        
        # If the range of valid r's is non-empty, add its size to the total.
        if upper_bound_r >= lower_bound_r:
            count_for_l = upper_bound_r - lower_bound_r + 1
            total_valid_pairs += count_for_l
            
    print(total_valid_pairs)

solve()