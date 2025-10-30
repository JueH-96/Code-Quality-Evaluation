import sys
import math
import random

# ---------- small prime table for trial division (factorisation of N) ----------
def sieve(limit: int):
    mark = bytearray(b'\x01') * (limit + 1)
    mark[0:2] = b'\x00\x00'
    for p in range(2, int(limit ** 0.5) + 1):
        if mark[p]:
            step = p * p
            mark[step: limit + 1: p] = b'\x00' * ((limit - step) // p + 1)
    return [i for i, v in enumerate(mark) if v]

SMALL_PRIMES = sieve(31623)       # primes up to sqrt(1e9)

# ---------- Miller–Rabin on 64 bit --------------------------------------------
def is_probable_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p

    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1

    # deterministic bases for 64-bit integers
    for a in (2, 3, 5, 7, 11, 13):
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True

# ---------- factorisation (N ≤ 1e9, c is tiny) --------------------------------
def factorise(x: int):
    factors = {}
    tmp = x
    for p in SMALL_PRIMES:
        if p * p > tmp:
            break
        if tmp % p == 0:
            cnt = 0
            while tmp % p == 0:
                tmp //= p
                cnt += 1
            factors[p] = cnt
    if tmp > 1:
        factors[tmp] = 1
    return factors      # dictionary prime -> exponent

def merge_factorisations(dict1, dict2):
    res = dict1.copy()
    for k, v in dict2.items():
        res[k] = res.get(k, 0) + v
    return res

# ---------- primitive root -----------------------------------------------------
def primitive_root(p: int, factor_list):
    # factor_list: list of distinct prime divisors of p-1
    g = 2
    while True:
        ok = True
        for q in factor_list:
            if pow(g, (p - 1) // q, p) == 1:
                ok = False
                break
        if ok:
            return g
        g += 1

# ---------- main ---------------------------------------------------------------
def solve_case(N: int):
    if N == 1:
        return 1, 1

    c = 1
    while True:
        p = c * N + 1
        if p > 10**18:
            # should never happen with given limits
            c = None
            break
        if is_probable_prime(p):
            break
        c += 1

    # factorisation of p-1 = c*N
    facN = factorise(N)
    facC = factorise(c)  # c is tiny, trial division is enough
    fac = merge_factorisations(facN, facC)
    prime_divisors = list(fac.keys())

    g = primitive_root(p, prime_divisors)
    A = pow(g, c, p)
    return A, p


def main():
    data = sys.stdin.read().strip().split()
    if not data:
        return
    t = int(data[0])
    out_lines = []
    for i in range(1, t + 1):
        N = int(data[i])
        A, M = solve_case(N)
        out_lines.append(f"{A} {M}")
    sys.stdout.write("
".join(out_lines))

if __name__ == "__main__":
    main()