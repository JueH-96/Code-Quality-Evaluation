# YOUR CODE HERE
import sys

def calculate_mex(a, b, c):
    """Calculates the minimum non-negative integer not in the set {a, b, c}."""
    s = {a, b, c}
    if 0 not in s:
        return 0
    if 1 not in s:
        return 1
    if 2 not in s:
        return 2
    return 3 # 0, 1, 2 are all present

def solve():
    # Read input
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    S = sys.stdin.readline().strip()

    # Precompute prefix counts of 'M' characters with specific values
    # prefix_M[j][v]: count of indices i < j such that S[i] == 'M' and A[i] == v
    # Use N+1 size to easily handle indices from 0 to N
    prefix_M = [[0] * 3 for _ in range(N + 1)]
    for j in range(1, N + 1):
        # Copy counts from the previous index
        for v in range(3):
            prefix_M[j][v] = prefix_M[j - 1][v]
        # If the character at the previous index was 'M', increment the count for its value
        if S[j - 1] == 'M':
            prefix_M[j][A[j - 1]] += 1

    # Precompute suffix counts of 'X' characters with specific values
    # suffix_X[j][v]: count of indices k >= j such that S[k] == 'X' and A[k] == v
    # Use N+1 size to easily handle indices from 0 to N+1
    suffix_X = [[0] * 3 for _ in range(N + 1)]
    # Iterate backwards from N-1 down to 0
    for j in range(N - 1, -1, -1):
        # Copy counts from the next index
        for v in range(3):
            suffix_X[j][v] = suffix_X[j + 1][v]
        # If the character at the current index is 'X', increment the count for its value
        if S[j] == 'X':
            suffix_X[j][A[j]] += 1

    total_mex_sum = 0

    # Iterate through all possible middle indices j where S[j] == 'E'
    # We need 0 <= i < j < k <= N-1
    # S[i] == 'M', S[j] == 'E', S[k] == 'X'
    # The loop over j from 0 to N-1 covers all indices. If S[j] is not 'E', we skip.
    # If j is too early (j=0) or too late (j=N-1), prefix_M[j] or suffix_X[j+1] will have zero counts,
    # implicitly handling the boundaries i<j and k>j.
    for j in range(N):
        if S[j] == 'E':
            v_j = A[j] # Value at the 'E' position

            # Iterate through all possible values for A[i] (v_i) at 'M' position (i < j)
            # and all possible values for A[k] (v_k) at 'X' position (k > j)
            for v_i in range(3): # v_i can be 0, 1, or 2
                for v_k in range(3): # v_k can be 0, 1, or 2
                    # Number of valid indices i before j with S[i] == 'M' and A[i] == v_i
                    # These are indices i in [0, j-1]. prefix_M[j] stores counts for indices up to j-1.
                    count_i = prefix_M[j][v_i]

                    # Number of valid indices k after j with S[k] == 'X' and A[k] == v_k
                    # These are indices k in [j+1, N-1]. suffix_X[j+1] stores counts for indices from j+1 up to N-1.
                    count_k = suffix_X[j + 1][v_k]

                    # Add the contribution: (count of i's) * (count of k's) * mex_value
                    # If count_i or count_k is 0, the product is 0, so this correctly handles cases
                    # where there are no valid i or k for the given j and values.
                    total_mex_sum += count_i * count_k * calculate_mex(v_i, v_j, v_k)

    # Print the total sum
    print(total_mex_sum)

solve()