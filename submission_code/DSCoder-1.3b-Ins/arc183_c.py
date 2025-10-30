import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    conditions = []
    for _ in range(M):
        L, R, X = map(int, sys.stdin.readline().split())
        conditions.append((L, R, X))
    conditions.sort()

    dp = [0] * (N + 1)
    dp[1] = 1
    for i in range(2, N + 1):
        dp[i] = (dp[i - 1] + dp[i - 2]) % 998244353

    ans = 0
    max_so_far = 0
    for L, R, X in conditions:
        if max_so_far < X:
            max_so_far = X
            ans = (ans + dp[R - L + 1]) % 998244353

    print(ans)

solve()