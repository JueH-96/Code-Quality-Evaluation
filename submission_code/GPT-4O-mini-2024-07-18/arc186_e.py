def count_sequences(N, M, K, X):
    MOD = 998244353

    # Create a DP table where dp[i][j] means the number of ways to form a sequence of length i
    # such that the forbidden subsequence X is not present.
    dp = [[0] * (M + 1) for _ in range(N + 1)]
    dp[0][0] = 1  # Base case: one way to have an empty sequence with no forbidden subsequence.

    for i in range(1, N + 1):
        for j in range(M + 1):
            # If we don't add the next element from the forbidden subsequence
            dp[i][j] = dp[i - 1][j] * (K - (1 if j == 0 else 0)) % MOD
            
            # If we are adding the next element from the forbidden subsequence
            if j > 0 and X[j - 1] <= K:
                dp[i][j] = (dp[i][j] + dp[i - 1][j - 1]) % MOD

    # The answer is the number of sequences of length N that do not contain the forbidden subsequence
    return dp[N][M]

import sys
input = sys.stdin.read
data = input().strip().split()
N, M, K = map(int, data[:3])
X = list(map(int, data[3:]))

result = count_sequences(N, M, K, X)
print(result)