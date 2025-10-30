import sys
import math

def solve():
    # Read input
    N = int(sys.stdin.readline())
    P = list(map(int, sys.stdin.readline().split()))

    # dp_prev stores max numerator sum using k-1 contests ending at index i
    # dp_curr stores max numerator sum using k contests ending at index i
    # Each is an array of size N, indexed 0 to N-1 corresponding to original contest indices

    # Initialize dp_prev for k=1
    # dp[1][i] = P[i] for i = 0 to N-1
    dp_prev = [float('-inf')] * N
    for i in range(N):
        dp_prev[i] = float(P[i]) # Store as float

    # Precompute powers of 0.9 and denominators
    # p09[m] = (0.9)^m
    # D[k] = sum_{m=0}^{k-1} (0.9)^m
    p09 = [0.0] * N
    D = [0.0] * (N + 1)
    p09[0] = 1.0
    D[1] = 1.0
    for m in range(1, N):
        p09[m] = p09[m-1] * 0.9
    for k in range(2, N + 1):
        D[k] = D[k-1] + p09[k-1]

    # Calculate initial max rating for k=1
    max_num_k = float('-inf')
    for i in range(N):
        max_num_k = max(max_num_k, dp_prev[i])

    max_overall_rating = float('-inf')
    if max_num_k != float('-inf'): # Should always be true for N >= 1 as P_i >= 1
        max_overall_rating = max(max_overall_rating, max_num_k / D[1] - 1200.0 / math.sqrt(1))

    # DP for k = 2 to N
    dp_curr = [float('-inf')] * N

    for k in range(2, N + 1):
        # max_so_far tracks max(dp_prev[j]) for k-2 <= j <= current_i - 1
        # This represents the maximum numerator sum using k-1 contests ending at any index j < i,
        # where j must be at least k-2 to accommodate k-1 contests.
        max_so_far = float('-inf')

        for i in range(k - 1, N): # P[i] is the k-th item. Index i must be at least k-1.
            # To compute dp_curr[i] (which is dp[k][i]), we use the recurrence:
            # dp[k][i] = P[i] + 0.9 * max(dp[k-1][j]) for k-2 <= j <= i-1.
            # The values dp[k-1][j] are stored in dp_prev[j].
            # We need max(dp_prev[j]) for j from k-2 up to i-1.
            # The value dp_prev[i-1] was computed when we processed k-1 items ending at index i-1.
            # This value is relevant for the max needed for dp[k][i].
            # It is valid if i-1 >= (k-1)-1 = k-2. This is true since i >= k-1.
            # We update max_so_far to include dp_prev[i-1] *before* using max_so_far to compute dp_curr[i].
            # This ensures max_so_far correctly reflects the maximum over j up to i-1 when we are at iteration i.

            # Update max_so_far to include dp_prev[i-1]
            # The index i-1 is the latest possible index for the (k-1)-th contest before index i.
            # dp_prev[j] is valid for j >= k-2.
            # i-1 >= k-2 is always true in this loop since i >= k-1.
            max_so_far = max(max_so_far, dp_prev[i - 1])

            # If a valid previous state exists (i.e., max_so_far is not -inf)
            if max_so_far != float('-inf'):
                 dp_curr[i] = float(P[i]) + 0.9 * max_so_far
            # else: dp_curr[i] remains -inf, meaning this state is unreachable

        # Find MaxNum[k] from dp_curr
        # MaxNum[k] is the maximum numerator sum choosing exactly k contests, ending at any index i.
        max_num_k = float('-inf')
        for i in range(k - 1, N):
             max_num_k = max(max_num_k, dp_curr[i])

        # Calculate and update max rating for this k
        if max_num_k != float('-inf'): # Check if k items are possible (numerator sum is not -inf)
             rating_k = max_num_k / D[k] - 1200.0 / math.sqrt(k)
             max_overall_rating = max(max_overall_rating, rating_k)

        # Prepare for next iteration (k+1): dp_prev becomes dp_curr
        # Reset dp_curr for the next k loop by swapping and re-initializing
        dp_prev, dp_curr = dp_curr, dp_prev
        # Reset dp_curr for the next iteration. Only indices >= k will be used,
        # but resetting the whole array is simpler and O(N).
        for i in range(N):
             dp_curr[i] = float('-inf')

    # Print the result with high precision
    # Using .17f to match sample precision and requirement (absolute or relative error at most 10^{-6}).
    print(f"{max_overall_rating:.17f}")

solve()