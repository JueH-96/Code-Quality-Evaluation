# YOUR CODE HERE
import sys
from math import gcd

MOD = 998244353

def solve(N, A):
    dp = [0] * (N + 1)
    sum_dp = [0] * (N + 1)
    
    for i in range(1, N + 1):
        dp[i] = sum_dp[i-1]
        if i > 1:
            for j in range(i-1, 0, -1):
                g = gcd(A[j-1], A[i-1])
                dp[i] = (dp[i] + g * pow(2, i-j-1, MOD)) % MOD
        sum_dp[i] = (sum_dp[i-1] + dp[i]) % MOD
        print(dp[i])

# Read input
N = int(sys.stdin.readline())
A = list(map(int, sys.stdin.readline().split()))

# Solve and print results
solve(N, A)