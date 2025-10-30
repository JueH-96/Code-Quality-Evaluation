# YOUR CODE HERE
def min_time_to_type(X, Y, Z, S):
    n = len(S)
    dp = [[float('inf')] * 2 for _ in range(n + 1)]
    dp[0][0] = 0  # Initial state with Caps Lock off
    dp[0][1] = Z  # Initial state with Caps Lock on (after pressing Caps Lock once)

    for i in range(n):
        if S[i] == 'a':
            dp[i + 1][0] = min(dp[i + 1][0], dp[i][0] + X)  # 'a' with Caps Lock off
            dp[i + 1][1] = min(dp[i + 1][1], dp[i][1] + Y)  # 'a' with Caps Lock on (Shift + a)
        else:  # S[i] == 'A'
            dp[i + 1][0] = min(dp[i + 1][0], dp[i][0] + Y)  # 'A' with Caps Lock off (Shift + a)
            dp[i + 1][1] = min(dp[i + 1][1], dp[i][1] + X)  # 'A' with Caps Lock on

        # Toggle Caps Lock
        dp[i + 1][0] = min(dp[i + 1][0], dp[i + 1][1] + Z)
        dp[i + 1][1] = min(dp[i + 1][1], dp[i + 1][0] + Z)

    print(min(dp[n][0], dp[n][1]))

import sys
input = sys.stdin.read
data = input().split()
X = int(data[0])
Y = int(data[1])
Z = int(data[2])
S = data[3]

min_time_to_type(X, Y, Z, S)