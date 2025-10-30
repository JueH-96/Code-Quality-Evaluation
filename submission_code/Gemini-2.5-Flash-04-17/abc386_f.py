import sys

def solve():
    # Read input
    K = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()

    N = len(S)
    M = len(T)

    # If difference in length is greater than K, it's impossible
    if abs(N - M) > K:
        print("No")
        return

    # DP state: dp[d_prime] stores the minimum cost to align S[0...i-1] with T[0...j-1]
    # where j = i - d and d_prime = d + K.
    # d = i - j ranges from -K to K for cells (i,j) within the band |i-j| <= K.
    # d_prime = d + K ranges from 0 to 2K.
    # We use two arrays of size 2*K + 1: dp (for row i-1) and next_dp (for row i).

    # Initialize DP table for row 0 (empty prefix of S)
    # dp[0][j] = j (cost to insert j characters of T) for 0 <= j <= M.
    # For row 0, i=0. j = 0 - d = -d. d' = K - j.
    # We only care about cells (0, j) where |0-j| <= K, i.e., j <= K.
    # Also needs 0 <= j <= M. So j is in [0, min(M, K)].
    dp = [K + 1] * (2 * K + 1) # Use K+1 to represent infinity
    inf = K + 1

    for j in range(min(M, K) + 1):
        d_prime = K - j
        # d_prime must be in [0, 2*K].
        # If j is in [0, min(M, K)]:
        # If M >= K, j in [0, K]. d_prime = K-j in [0, K]. Valid indices.
        # If M < K, j in [0, M]. d_prime = K-j in [K-M, K]. K-M >= K - (K-1) = 1 >= 0. Valid indices.
        dp[d_prime] = j

    # Iterate through rows i from 1 to N
    for i in range(1, N + 1):
        next_dp = [inf] * (2 * K + 1)

        # Determine the relevant range of d_prime for the current row i.
        # j = i - d = i - (d_prime - K).
        # We need 0 <= j <= M.
        # 0 <= i - d_prime + K <= M
        # d_prime <= i + K  and  d_prime >= i - M + K.
        # Also 0 <= d_prime <= 2*K.
        # So d_prime is in [max(0, i - M + K), min(2*K, i + K)].
        min_d_prime = max(0, i - M + K)
        max_d_prime = min(2 * K, i + K)

        # Iterate d_prime downwards to use next_dp[d_prime + 1] (for insert operation)
        # which will have been computed in the same row iteration.
        for d_prime in range(max_d_prime, min_d_prime - 1, -1):
            d = d_prime - K # Actual difference i - j
            j = i - d       # Current column index j. Guaranteed 0 <= j <= M by d_prime bounds.

            # Cost from deleting S[i-1]: from cell (i-1, j), difference d-1.
            # Previous d_prime index is d_prime - 1.
            cost_delete = inf
            if d_prime >= 1: # Check if d_prime - 1 is a valid index [0, 2K]
                 cost_delete = dp[d_prime - 1] + 1

            # Cost from inserting T[j-1]: from cell (i, j-1), difference d+1.
            # Current d_prime index is d_prime + 1.
            # Needs j > 0 for cell (i, j-1) to be valid.
            cost_insert = inf
            if j > 0 and d_prime <= 2 * K - 1: # Check if d_prime + 1 is a valid index [0, 2K]
                cost_insert = next_dp[d_prime + 1] + 1

            # Cost from replacing S[i-1] with T[j-1] (or match): from cell (i-1, j-1), difference d.
            # Previous d_prime index is d_prime.
            # Needs j > 0 for cell (i-1, j-1) to be valid.
            cost_replace = inf
            if j > 0:
                 # dp[d_prime] is the cost for (i-1, j-1)
                 cost_replace = dp[d_prime] + (0 if S[i-1] == T[j-1] else 1)

            # Calculate the minimum cost for cell (i, j) corresponding to d_prime
            next_dp[d_prime] = min(cost_delete, cost_insert, cost_replace)

        # Update dp to the newly computed next_dp for the next row iteration
        dp = next_dp

    # The final answer is the cost at cell (N, M).
    # Difference d = N - M.
    # Corresponding d_prime = N - M + K.
    target_d_prime = N - M + K

    # We already checked abs(N - M) <= K, which means 0 <= N - M + K <= 2*K.
    # So target_d_prime is guaranteed to be a valid index [0, 2K].

    # Check if the minimum cost to reach (N, M) is within K.
    if dp[target_d_prime] <= K:
        print("Yes")
    else:
        # This case should only happen if dp[target_d_prime] > K,
        # since we already handled abs(N-M) > K.
        print("No")

solve()