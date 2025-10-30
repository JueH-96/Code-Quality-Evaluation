# YOUR CODE HERE
MOD = 998244353

def count_valid_sequences(N, Q, operations):
    dp = [[0] * (N + 1) for _ in range(Q + 1)]
    dp[0][0] = 1

    for i in range(1, Q + 1):
        P_i, V_i = operations[i - 1]
        for j in range(N + 1):
            if dp[i - 1][j] == 0:
                continue
            # Option 1: Replace S_1 to S_P_i with V_i
            if all(dp[i - 1][k] <= V_i for k in range(P_i)):
                dp[i][P_i] = (dp[i][P_i] + dp[i - 1][j]) % MOD
            # Option 2: Replace S_P_i to S_N with V_i
            if all(dp[i - 1][k] <= V_i for k in range(P_i, N + 1)):
                dp[i][N] = (dp[i][N] + dp[i - 1][j]) % MOD

    return sum(dp[Q]) % MOD

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
Q = int(data[1])
operations = [(int(data[2 + 2 * i]), int(data[3 + 2 * i])) for i in range(Q)]

print(count_valid_sequences(N, Q, operations))