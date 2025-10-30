# YOUR CODE HERE
MOD = 998244353

def mod_inverse(a, m):
    m0, x0, x1 = m, 0, 1
    if m == 1:
        return 0
    while a > 1:
        q = a // m
        m, a = a % m, m
        x0, x1 = x1 - q * x0, x0
    if x1 < 0:
        x1 += m0
    return x1

def expected_salary(N, A):
    dp = [0] * (N + 1)
    total = 0
    for i in range(N, 0, -1):
        dp[i] = (A[i-1] + total) * mod_inverse(i, MOD)
        total = (total + dp[i]) % MOD
    return total

import sys
input = sys.stdin.read().split()
N = int(input[0])
A = list(map(int, input[1:]))

print(expected_salary(N, A))