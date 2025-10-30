import sys

def solve():
    """
    Reads inputs, calculates the banded Levenshtein distance, and prints the result.
    """
    try:
        K = int(sys.stdin.readline())
        S = sys.stdin.readline().strip()
        T = sys.stdin.readline().strip()
    except (IOError, ValueError):
        return

    N = len(S)
    M = len(T)

    # If the length difference is greater than K, it's impossible.
    if abs(N - M) > K:
        print("No")
        return

    # Optimization: ensure S is the shorter or equal length string to minimize
    # the number of iterations in the main loop. Levenshtein distance is symmetric.
    if N > M:
        S, T = T, S
        N, M = M, N

    # We use a banded DP approach. We only need to store two rows of the band.
    # `prev_dp[d + K]` stores the cost for diagonal d = j - i for the previous row.
    # A value of K + 1 acts as infinity, since we only care about distances <= K.
    inf = K + 1
    prev_dp = [inf] * (2 * K + 1)

    # Initialize for row i = 0 (transforming "" to prefixes of T).
    # The Levenshtein distance dp[0][j] is j. This corresponds to diagonal d = j.
    for d in range(K + 1):
        if d <= M:
            prev_dp[d + K] = d

    # Iterate through rows of the conceptual DP table (for each character in S).
    for i in range(1, N + 1):
        curr_dp = [inf] * (2 * K + 1)
        
        # Base case for the current row: dp[i][0] = i (deleting i chars from S).
        # This corresponds to diagonal d = 0 - i = -i.
        d_base = -i
        if -K <= d_base <= K:
            curr_dp[d_base + K] = i

        # Loop through the diagonals in the band. d = j - i.
        for d in range(-K, K + 1):
            d_idx = d + K
            j = i + d
            
            # Skip if j is out of bounds for string T or not in the main recurrence (j>=1).
            if not (1 <= j <= M):
                continue

            # Cost of substitution (1) or match (0).
            cost = 0 if S[i - 1] == T[j - 1] else 1

            # Option 1: Match or substitute from dp[i-1][j-1].
            val_match_sub = prev_dp[d_idx] + cost

            # Option 2: Deletion from S, from dp[i-1][j].
            val_del = inf
            if d + 1 <= K:
                val_del = prev_dp[d_idx + 1] + 1
            
            # Option 3: Insertion into S, from dp[i][j-1].
            val_ins = inf
            if d - 1 >= -K:
                val_ins = curr_dp[d_idx - 1] + 1

            curr_dp[d_idx] = min(val_match_sub, val_del, val_ins)
        
        prev_dp = curr_dp

    # The final edit distance is at dp[N][M].
    # The diagonal is d = M - N.
    final_d = M - N
    final_dist = prev_dp[final_d + K]

    if final_dist <= K:
        print("Yes")
    else:
        print("No")

solve()