# YOUR CODE HERE
import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    wheels = []
    for _ in range(N):
        C, P, *S = map(int, sys.stdin.readline().split())
        wheels.append((C, P, S))

    dp = [0] * (M + 1)
    for i in range(1, M + 1):
        dp[i] = min(dp[i - j] + C for C, P, S in wheels for j in S if i - j >= 0)

    print(dp[-1])

solve()