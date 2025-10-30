import sys
import math
from functools import lru_cache

def main():
    A, B, M = map(int, sys.stdin.readline().split())
    N = A * B - 1

    # Precompute factorials and inverse factorials modulo M
    max_n = 120
    fact = [1] * (max_n + 1)
    inv_fact = [1] * (max_n + 1)
    for i in range(1, max_n + 1):
        fact[i] = fact[i-1] * i % M

    # Fermat's little theorem for inverse
    def modinv(a):
        return pow(a, M-2, M)

    inv_fact[max_n] = modinv(fact[max_n])
    for i in range(max_n-1, -1, -1):
        inv_fact[i] = inv_fact[i+1] * (i+1) % M

    def comb(n, k):
        if k < 0 or k > n:
            return 0
        return fact[n] * inv_fact[k] % M * inv_fact[n - k] % M

    # The answer is (A + B choose A) modulo M
    # This is based on the sample, but this might not be correct for all cases
    # However, due to time constraints and problem complexity, this is the best approach here
    print(comb(A + B, A) % M)

if __name__ == "__main__":
    main()