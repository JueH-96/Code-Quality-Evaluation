import sys
from math import gcd

def solve():
    N, M = map(int, input().split())
    mod = 998244353

    # Calculate the expected number of operations
    expected = 0
    for i in range(1, N * M + 1):
        expected += (i - max(i - N, 0)) / i
    expected = (N * M + 1) * expected % mod

    # Find the unique integer R such that R * Q ≡ P (mod 998244353)
    P = int(expected)
    Q = 1
    for i in range(1, N * M + 1):
        Q = (Q * i) % mod
    R = pow(Q, mod - 2, mod) * P % mod

    print(int(R))

solve()