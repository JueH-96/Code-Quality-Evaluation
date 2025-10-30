import sys
import threading
import random
def main():
    import sys
    sys.setrecursionlimit(1000000)
    data = sys.stdin.read().split()
    T = int(data[0])
    Ns = list(map(int, data[1:]))

    # Miller-Rabin primality test for n < 2^64
    def is_prime(n):
        if n < 2:
            return False
        # small primes
        small_primes = [2,3,5,7,11,13,17,19,23,29]
        for p in small_primes:
            if n % p == 0:
                return n == p
        # write n-1 as d*2^s
        d = n-1
        s = 0
        while d & 1 == 0:
            d >>= 1
            s += 1
        # bases for deterministic test <=2^64
        for a in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
            if a % n == 0:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n-1:
                continue
            for _ in range(s-1):
                x = (x * x) % n
                if x == n-1:
                    break
            else:
                return False
        return True

    # Pollard's Rho for factorization
    def pollard_rho(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        # random c, f(x)=x^2+c
        while True:
            x = random.randrange(2, n-1)
            y = x
            c = random.randrange(1, n-1)
            d = 1
            while d == 1:
                x = (x*x + c) % n
                y = (y*y + c) % n
                y = (y*y + c) % n
                d = math.gcd(abs(x-y), n)
                if d == n:
                    break
            if d > 1 and d < n:
                return d

    import math
    # Full factorization
    def factor(n, res):
        if n == 1:
            return
        if is_prime(n):
            res.append(n)
        else:
            d = pollard_rho(n)
            factor(d, res)
            factor(n//d, res)

    # cache factorizations of N to speed up repeats
    fact_cache = {}

    for N in Ns:
        if N == 1:
            # minimal exponent 1: any A>1, M=1 works
            print(2, 1)
            continue
        # find k so that p = k*N + 1 is prime
        # limit search
        for k in range(1, 1001):
            p = k * N + 1
            if is_prime(p):
                break
        else:
            # fallback (shouldn't happen per problem)
            k = 1
            p = N + 1
            # assume prime
        # factor p-1 = k*N
        # factor N
        if N in fact_cache:
            facN = fact_cache[N]
        else:
            tmp = []
            factor(N, tmp)
            facN = {}
            for q in tmp:
                facN[q] = facN.get(q, 0) + 1
            fact_cache[N] = facN
        # factor k by trial
        facK = {}
        kk = k
        # trial small primes
        for pr in (2,3,5,7,11,13,17,19,23,29,31,37,41,43,47):
            while kk % pr == 0:
                facK[pr] = facK.get(pr,0)+1
                kk //= pr
        # if remainder >1
        if kk > 1:
            if is_prime(kk):
                facK[kk] = facK.get(kk,0)+1
            else:
                tmp2 = []
                factor(kk, tmp2)
                for q in tmp2:
                    facK[q] = facK.get(q,0)+1
        # collect unique prime factors of p-1
        pf = list(facN.keys()) + list(facK.keys())
        # find a primitive root modulo p
        # primitive root g satisfies for each prime q|p-1: g^((p-1)/q) mod p != 1
        for g in range(2, 1<<20):
            ok = True
            for q in pf:
                if pow(g, (p-1)//q, p) == 1:
                    ok = False
                    break
            if ok:
                root = g
                break
        # element of order N is g^k mod p, since order(g^k) = (p-1)/gcd(p-1, k) = k*N/k = N
        A = pow(root, k, p)
        print(A, p)

if __name__ == "__main__":
    main()