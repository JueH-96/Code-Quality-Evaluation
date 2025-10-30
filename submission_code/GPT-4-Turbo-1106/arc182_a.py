MOD = 998244353

def count_sequences(N, Q, operations):
    # Initialize dp array
    dp = [[0] * (N + 1) for _ in range(Q + 1)]
    dp[0][N] = 1  # Base case: all elements are 0, so only one way to start

    for i in range(1, Q + 1):
        P, V = operations[i - 1]
        # Calculate prefix sums for optimization
        prefix_sum = [0] * (N + 2)
        for j in range(N + 1):
            prefix_sum[j + 1] = (prefix_sum[j] + dp[i - 1][j]) % MOD

        for j in range(1, N + 1):
            if j < P:
                # If we replace S_1 to S_j with V, then S_{j+1} to S_N can be anything
                # that is not greater than V. Thus, we sum up all possibilities from
                # dp[i-1][j] to dp[i-1][N] where the values are <= V.
                dp[i][j] = prefix_sum[N + 1] - prefix_sum[j]
            else:
                # If we replace S_j to S_N with V, then S_1 to S_{j-1} must be <= V.
                # We sum up all possibilities from dp[i-1][0] to dp[i-1][j-1].
                dp[i][j] = prefix_sum[j]

            # Since we're dealing with differences, we might get a negative number.
            # We need to ensure the result is positive and within the range [0, MOD).
            dp[i][j] = (dp[i][j] + MOD) % MOD

    # The answer is the sum of all dp[Q][j] where 1 <= j <= N
    return sum(dp[Q][1:]) % MOD

# Read input
N, Q = map(int, input().split())
operations = [tuple(map(int, input().split())) for _ in range(Q)]

# Solve the problem and output the answer
print(count_sequences(N, Q, operations))