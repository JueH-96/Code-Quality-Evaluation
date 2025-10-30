# YOUR CODE HERE
import sys

def ceil_div(a, b):
    return (a + b - 1) // b

def solve():
    N, X, Y = map(int, input().split())
    buses = [tuple(map(int, input().split())) for _ in range(N-1)]
    Q = int(input())
    queries = [int(input()) for _ in range(Q)]

    dp = [0] * N
    for q in queries:
        dp[0] = q + X
        for i in range(1, N):
            P, T = buses[i-1]
            dp[i] = ceil_div(dp[i-1], P) * P + T
        print(dp[N-1] + Y)

solve()