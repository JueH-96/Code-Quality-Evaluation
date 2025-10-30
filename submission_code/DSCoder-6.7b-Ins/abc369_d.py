import sys

def solve():
    N = int(sys.stdin.readline().strip())
    A = list(map(int, sys.stdin.readline().strip().split()))

    dp = [0] * (N + 1)
    dp[0] = 0
    dp[1] = A[0]

    for i in range(2, N + 1):
        dp[i] = max(dp[i - 1], dp[i - 2] + A[i - 1])

    print(dp[N])

solve()