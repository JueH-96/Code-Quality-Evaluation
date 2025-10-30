import sys
import math
from math import comb

MOD = 998244353

def main():
    N, M = map(int, sys.stdin.readline().split())

    # Precompute smallest prime factors up to sqrt(N)
    max_spf = int(math.isqrt(N)) + 1
    spf = list(range(max_spf))
    for i in range(2, int(math.isqrt(max_spf)) + 1):
        if spf[i] == i:
            for j in range(i*i, max_spf, i):
                if spf[j] == j:
                    spf[j] = i

    def factorize(x):
        factors = {}
        while x > 1:
            p = spf[x]
            while x % p == 0:
                if p in factors:
                    factors[p] += 1
                else:
                    factors[p] = 1
                x //= p
            if x > 1:
                if spf[x] == x:
                    p = x
                else:
                    p = spf[x]
                while x % p == 0:
                    if p in factors:
                        factors[p] += 1
                    else:
                        factors[p] = 1
                    x //= p
        return list(factors.items())

    total = 0

    for x in range(1, N + 1):
        if x == 1:
            continue
        factors = factorize(x)
        product = 1
        for (p, e) in factors:
            if p == 3:
                continue
            if p % 3 == 1:
                term = (e + 1) % 3
            else:
                if e % 2 == 1:
                    term = 0
                else:
                    term = 1
            product = (product * term) % 3
            if product == 0:
                break
        if product == 0:
            ways = 1
            for (p, e) in factors:
                ways = (ways * comb(e + M - 1, M - 1)) % MOD
            total = (total + ways) % MOD

    print(total % MOD)

if __name__ == "__main__":
    main()