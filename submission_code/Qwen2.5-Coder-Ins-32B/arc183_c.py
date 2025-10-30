# YOUR CODE HERE
MOD = 998244353

def mod_inverse(x, mod):
    return pow(x, mod - 2, mod)

def nCr(n, r, mod):
    if r > n:
        return 0
    if r == 0 or r == n:
        return 1
    numerator = 1
    denominator = 1
    for i in range(r):
        numerator = numerator * (n - i) % mod
        denominator = denominator * (i + 1) % mod
    return numerator * mod_inverse(denominator, mod) % mod

def solve(N, M, conditions):
    dp = [0] * (1 << N)
    dp[0] = 1
    for mask in range(1 << N):
        cnt = bin(mask).count('1')
        for i in range(N):
            if mask & (1 << i):
                valid = True
                for L, R, X in conditions:
                    L -= 1
                    R -= 1
                    X -= 1
                    if L <= i <= R and i == X:
                        valid = False
                        break
                    if L <= i <= R:
                        max_index = -1
                        for j in range(L, R + 1):
                            if mask & (1 << j):
                                if max_index == -1 or (mask >> j) & 1:
                                    max_index = j
                        if max_index == X:
                            valid = False
                            break
                if valid:
                    dp[mask] = (dp[mask] + dp[mask ^ (1 << i)]) % MOD
    return dp[(1 << N) - 1]

import sys
input = sys.stdin.read().split()
N = int(input[0])
M = int(input[1])
conditions = []
index = 2
for _ in range(M):
    L = int(input[index])
    R = int(input[index + 1])
    X = int(input[index + 2])
    conditions.append((L, R, X))
    index += 3

print(solve(N, M, conditions))