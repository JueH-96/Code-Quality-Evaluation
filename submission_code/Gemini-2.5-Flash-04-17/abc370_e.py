import sys

def solve():
    # Read input
    N, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))

    # Modulo constant
    MOD = 998244353

    # Compute prefix sums
    # P[i] will store the sum of A[0]...A[i-1]. P[0] = 0.
    P = [0] * (N + 1)
    for i in range(N):
        P[i+1] = P[i] + A[i]

    # Dynamic programming
    # dp[i] = number of ways to divide A[0...i-1] into contiguous subsequences
    #         such that none of these subsequences sum to K.
    # dp[0] = 1 (representing the valid empty division of an empty prefix)

    # To compute dp[i], we consider the last subsequence A[j...i-1] (0 <= j < i).
    # The number of valid divisions of A[0...i-1] ending with A[j...i-1] is dp[j]
    # if the sum A[j...i-1] != K.
    # The sum A[j...i-1] is P[i] - P[j].
    # dp[i] = sum_{j=0}^{i-1} dp[j] * [P[i] - P[j] != K]
    # dp[i] = sum_{j=0}^{i-1} dp[j] - sum_{j=0}^{i-1} dp[j] * [P[i] - P[j] == K]
    # dp[i] = sum_{j=0}^{i-1} dp[j] - sum_{j=0}^{i-1} dp[j] * [P[j] == P[i] - K]

    # We need sum(dp[j] for j < i) and sum(dp[j] for j < i and P[j] == V) for some V.
    # Let total_dp_sum_so_far = sum_{j=0}^{i-1} dp[j]
    # Let value_to_dp_sum_so_far[v] = sum_{j=0}^{i-1} dp[j] * [P[j] == v]

    # Initialization for i=0 (empty prefix):
    # dp[0] = 1
    total_dp_sum_so_far = 1
    value_to_dp_sum_so_far = {P[0]: 1} # P[0]=0, dp[0]=1

    # dp_i will store the current dp[i] value in each iteration.
    # After the loop, it will hold the final answer dp[N].
    dp_i = 0

    # Iterate from i=1 to N to compute dp[1], dp[2], ..., dp[N]
    for i in range(1, N + 1):
        # Calculate the sum of dp[j] for j < i where P[j] == P[i] - K
        target_prefix_sum = P[i] - K
        forbidden_sum = value_to_dp_sum_so_far.get(target_prefix_sum, 0)

        # Calculate dp[i] = total_dp_sum_so_far - forbidden_sum
        dp_i = (total_dp_sum_so_far - forbidden_sum + MOD) % MOD

        # After computing dp[i], update sums for the next iteration (i+1)
        # total_dp_sum_so_far will become sum_{j=0}^{i} dp[j]
        total_dp_sum_so_far = (total_dp_sum_so_far + dp_i) % MOD
        # value_to_dp_sum_so_far needs to include dp[i] for P[i]
        value_to_dp_sum_so_far[P[i]] = (value_to_dp_sum_so_far.get(P[i], 0) + dp_i) % MOD

    # After the loop finishes, dp_i holds the value of dp[N]
    dp_N = dp_i

    # The final answer is dp[N]
    print(dp_N)

solve()