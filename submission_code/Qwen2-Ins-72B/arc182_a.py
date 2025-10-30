import sys
from collections import deque

MOD = 998244353

def main():
    N, Q = map(int, sys.stdin.readline().split())
    P = [0] * Q
    V = [0] * Q
    for i in range(Q):
        P[i], V[i] = map(int, sys.stdin.readline().split())

    # dp[i][j] := the number of sequences of operations up to i-th operation
    # where the maximum value in the sequence is j
    dp = [[0] * (Q + 1) for _ in range(Q + 1)]
    dp[0][0] = 1

    # max_v[i] := the maximum value in the sequence up to i-th operation
    max_v = [0] * (Q + 1)

    for i in range(Q):
        p, v = P[i], V[i]
        max_v[i + 1] = max(max_v[i], v)

        # update dp[i+1][j] for j < v
        for j in range(max_v[i]):
            dp[i + 1][j] = dp[i][j]

        # update dp[i+1][v]
        dp[i + 1][v] = sum(dp[i][j] for j in range(v)) % MOD

        # update dp[i+1][j] for j > v
        for j in range(v + 1, max_v[i + 1]):
            dp[i + 1][j] = (dp[i][j] + dp[i + 1][v]) % MOD

    # calculate the number of sequences of operations where Snuke can perform all operations without crying
    ans = 0
    for j in range(max_v[-1] + 1):
        ans = (ans + dp[-1][j]) % MOD

    print(ans)

if __name__ == "__main__":
    main()