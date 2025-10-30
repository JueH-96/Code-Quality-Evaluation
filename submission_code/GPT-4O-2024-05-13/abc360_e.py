# YOUR CODE HERE
MOD = 998244353

def modinv(a, p):
    return pow(a, p - 2, p)

def solve(N, K):
    if N == 1:
        print(1)
        return
    
    invN = modinv(N, MOD)
    invN1 = modinv(N - 1, MOD)
    
    expected_value = 1
    for _ in range(K):
        expected_value = (expected_value * (N - 2) * invN1 + 1) % MOD
    
    expected_value = (expected_value * (N + 1) * invN) % MOD
    print(expected_value)

import sys
input = sys.stdin.read
N, K = map(int, input().strip().split())
solve(N, K)