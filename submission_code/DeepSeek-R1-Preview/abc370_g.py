import sys
import math

MOD = 998244353

def main():
    N, M = map(int, sys.stdin.readline().split())

    def comb_mod(n, k, MOD):
        if n < k or k < 0:
            return 0
        if k == 0 or n == k:
            return 1
        k = min(k, n - k)
        numerator = 1
        for i in range(n, n - k, -1):
            numerator = numerator * i % MOD
        denominator = 1
        for i in range(1, k + 1):
            denominator = denominator * i % MOD
        return numerator * pow(denominator, MOD - 2, MOD) % MOD

    def sum_comb(m, e):
        res = 0
        for k in range(e + 1):
            res = (res + comb_mod(k + m - 1, m - 1, MOD)) % MOD
        return res

    def sum_comb_mod1(m, e):
        res = 0
        for k in range(e + 1):
            if k % 3 != 2:
                res = (res + comb_mod(k + m - 1, m - 1, MOD)) % MOD
        return res

    def sum_comb_mod2(m, e):
        res = 0
        for k in range(0, e + 1, 2):
            res = (res + comb_mod(k + m - 1, m - 1, MOD)) % MOD
        return res

    def log2(n):
        return n.bit_length() - 1

    max_p = int(N ** 0.5) + 2
    sieve = [True] * (max_p + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(max_p ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, max_p + 1, i):
                sieve[j] = False
    primes = [i for i, is_prime in enumerate(sieve) if is_prime]

    contrib_3 = 1
    e3 = 0
    p = 3
    while p <= N:
        e3 += 1
        p *= 3
    e3 -= 1
    contrib_3 = sum_comb(M, e3)

    contrib_1 = 1
    contrib_2 = 1

    for p in primes:
        if p == 3:
            continue
        if p > N:
            break
        if p % 3 == 1:
            e = 0
            pe = 1
            while pe * p <= N:
                e += 1
                pe *= p
            s = sum_comb_mod1(M, e)
            contrib_1 = contrib_1 * s % MOD
        elif p % 3 == 2:
            e = 0
            pe = 1
            while pe * p <= N:
                e += 1
                pe *= p
            s = sum_comb_mod2(M, e)
            contrib_2 = contrib_2 * s % MOD

    U = contrib_3 * contrib_1 % MOD
    U = U * contrib_2 % MOD

    T_contrib_3 = sum_comb(M, e3)
    T_contrib_1 = 1
    T_contrib_2 = 1

    for p in primes:
        if p == 3:
            continue
        if p > N:
            break
        e = 0
        pe = 1
        while pe * p <= N:
            e += 1
            pe *= p
        s = sum_comb(M, e)
        if p % 3 == 1:
            T_contrib_1 = T_contrib_1 * s % MOD
        else:
            T_contrib_2 = T_contrib_2 * s % MOD

    T = T_contrib_3 * T_contrib_1 % MOD
    T = T * T_contrib_2 % MOD

    ans = (T - U) % MOD
    print(ans)

if __name__ == '__main__':
    main()