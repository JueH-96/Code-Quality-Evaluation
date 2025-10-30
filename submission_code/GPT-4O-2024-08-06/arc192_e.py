# YOUR CODE HERE
import sys
input = sys.stdin.read

MOD = 998244353

def modinv(a, p):
    return pow(a, p - 2, p)

def precompute_factorials(max_n, mod):
    fact = [1] * (max_n + 1)
    inv_fact = [1] * (max_n + 1)
    for i in range(2, max_n + 1):
        fact[i] = fact[i - 1] * i % mod
    inv_fact[max_n] = modinv(fact[max_n], mod)
    for i in range(max_n - 1, 0, -1):
        inv_fact[i] = inv_fact[i + 1] * (i + 1) % mod
    return fact, inv_fact

def comb(n, k, fact, inv_fact, mod):
    if n < k or k < 0:
        return 0
    return fact[n] * inv_fact[k] % mod * inv_fact[n - k] % mod

def solve():
    W, H, L, R, D, U = map(int, input().strip().split())
    
    max_factorial = W + H
    fact, inv_fact = precompute_factorials(max_factorial, MOD)
    
    total_paths = comb(W + H, W, fact, inv_fact, MOD)
    
    forbidden_paths = 0
    
    for x in range(L):
        for y in range(D):
            forbidden_paths += comb(x + y, x, fact, inv_fact, MOD) * comb(W - x + H - y, W - x, fact, inv_fact, MOD)
            forbidden_paths %= MOD
    
    for x in range(R + 1, W + 1):
        for y in range(U + 1, H + 1):
            forbidden_paths += comb(x + y, x, fact, inv_fact, MOD) * comb(W - x + H - y, W - x, fact, inv_fact, MOD)
            forbidden_paths %= MOD
    
    result = (total_paths - forbidden_paths + MOD) % MOD
    print(result)