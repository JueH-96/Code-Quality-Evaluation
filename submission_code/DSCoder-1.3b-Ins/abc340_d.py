import sys

def solve():
    N = int(sys.stdin.readline().strip())
    A = list(map(int, sys.stdin.readline().strip().split()))
    B = list(map(int, sys.stdin.readline().strip().split()))
    X = list(map(int, sys.stdin.readline().strip().split()))

    dp = [0] * (N + 1)
    for i in range(1, N + 1):
        dp[i] = min(dp[i - 1] + A[i - 1], dp[X[i - 1]] + B[i - 1])

    print(dp[N])

solve()