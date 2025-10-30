import sys

def solve():
    N, W = map(int, sys.stdin.readline().split())
    items = [tuple(map(int, sys.stdin.readline().split())) for _ in range(N)]

    dp = [0] * (W + 1)
    for w, v in items:
        for i in range(w, W + 1):
            dp[i] = max(dp[i], dp[i - w] + v - w**2)

    print(dp[-1])

solve()