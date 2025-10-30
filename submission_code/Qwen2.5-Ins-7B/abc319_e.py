# YOUR CODE HERE
from math import gcd
from functools import reduce

def lcm(a, b):
    return a * b // gcd(a, b)

def solve(N, X, Y, P, T, Q, q):
    dp = [float('inf')] * (10**9 + 1)
    dp[0] = X
    for i in range(N-1):
        for t in range(T[i]):
            if dp[t] == float('inf'):
                continue
            for j in range(P[i], 10**9 + 1, P[i]):
                dp[j] = min(dp[j], dp[t] + j - t)
    for i in range(N-1):
        for t in range(T[i]):
            if dp[t] == float('inf'):
                continue
            for j in range(P[i], 10**9 + 1, P[i]):
                dp[j] = min(dp[j], dp[t] + j - t + T[i])
    for i in range(Q):
        print(dp[q[i]] + Y)