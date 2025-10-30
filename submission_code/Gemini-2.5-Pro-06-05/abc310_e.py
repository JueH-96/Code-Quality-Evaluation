import sys

def solve():
    """
    Reads input, solves the problem using an O(N) dynamic programming approach,
    and prints the result.
    """
    # Use sys.stdin.readline for faster I/O with large N
    try:
        N = int(sys.stdin.readline())
        S = sys.stdin.readline().strip()
    except (IOError, ValueError):
        # Handle cases of empty or malformed input
        return

    total_sum = 0
    # o_prev stores the count of starting indices i <= j-1 for which
    # the nested NAND evaluation f(i, j-1) results in 1.
    # This corresponds to O_{j-1} in a 0-indexed system.
    o_prev = 0

    # Iterate j from 0 to N-1. For each j, we calculate the contribution
    # from all subarrays ending at j. This contribution, o_curr, is the
    # number of starting indices i <= j for which f(i, j) = 1.
    for j in range(N):
        o_curr = 0
        
        if S[j] == '0':
            # If A_j is 0:
            # f(j, j) = 0.
            # For i < j, f(i, j) = f(i, j-1) NAND 0 = 1.
            # There are j such indices i (0 to j-1). So, the count of 1s is j.
            o_curr = j
        else: # S[j] == '1'
            # If A_j is 1:
            # f(j, j) = 1.
            # For i < j, f(i, j) = NOT(f(i, j-1)).
            # The number of 1s from i < j is the number of 0s from the previous step.
            # Number of 0s at step j-1 = (total indices up to j-1) - (number of 1s)
            #                            = j - o_prev
            # Total 1s at step j = (number of 0s from prev) + 1 (for f(j,j))
            o_curr = (j - o_prev) + 1

        total_sum += o_curr
        o_prev = o_curr
        
    print(total_sum)

solve()