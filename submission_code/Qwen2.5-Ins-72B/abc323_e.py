# YOUR CODE HERE
import sys
from math import gcd

MOD = 998244353

def modinv(a, m):
    a, b, u, v = a, m, 1, 0
    while a != 0:
        q, r = b // a, b % a
        m, n = u, v
        u, v = n - q * m, m
        b, a = a, r
    return (u + m) % m

def solve(N, X, T):
    total_time = sum(T)
    prob = 0
    for i in range(X + 2):
        if i * T[0] <= X + 1 and (i + 1) * T[0] > X + 1:
            prob += pow(N, -i, MOD)
    prob = prob * pow(N, MOD - 2, MOD) % MOD
    return prob

input = sys.stdin.read
data = input().split()
N, X = int(data[0]), int(data[1])
T = list(map(int, data[2:]))

print(solve(N, X, T))