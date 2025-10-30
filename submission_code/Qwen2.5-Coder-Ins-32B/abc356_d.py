# YOUR CODE HERE
MOD = 998244353

def popcount(x):
    return bin(x).count('1')

def sum_popcount_and(N, M):
    total = 0
    for k in range(N + 1):
        total = (total + popcount(k & M)) % MOD
    return total

import sys
input = sys.stdin.read().split()
N = int(input[0])
M = int(input[1])

print(sum_popcount_and(N, M))