import sys
import threading
import random
def main():
    import sys
    sys.setrecursionlimit(1000000)
    data = sys.stdin.read().split()
    T = int(data[0])
    Ns = list(map(int, data[1:]))

    # Miller–Rabin primality test
    def is_prime(n):
        if n < 2: return False
        # small primes
        small_primes = [2,3,5,7,11,13,17,19,23,29]
        for p in small_primes:
            if n == p: return True
            if n % p == 0: return False
        # write n-1 as d*2^s
        d = n-1
        s = 0
        while d & 1 == 0:
            d >>= 1
            s += 1
        # test bases
        for a in [2, 325, 9375, 28178, 450775, 9780504, 1795265022]:
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

    # Pollard's Rho
    def pollard_rho(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        if n % 5 == 0:
            return 5
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
    # factorization via Pollard
    def factor(n, res):
        if n == 1:
            return
        if is_prime(n):
            res.append(n)
        else:
            d = pollard_rho(n)
            factor(d, res)
            factor(n//d, res)

    out = []
    for N in Ns:
        if N == 1:
            # 2^1 -1 =1, minimal n=1
            out.append("2 1")
            continue
        # small N case: A=2, M=2^N -1 if it fits
        if N <= 60:
            M = (1 << N) - 1
            out.append(f"2 {M}")
            continue
        # general: find p = k*N + 1 prime
        found = False
        for k in range(1, 500):
            p = k * N + 1
            if p > 0 and is_prime(p):
                # factor p-1 = k*N
                facs = []
                factor(k, facs)
                factor(N, facs)
                # get unique prime factors
                primes = list(set(facs))
                # find primitive root g mod p
                for g in range(2, min(p, 1000)):
                    ok = True
                    for q in primes:
                        if pow(g, (p-1)//q, p) == 1:
                            ok = False
                            break
                    if ok:
                        # element of order p-1; we want A = g^k, order = (p-1)/gcd(p-1,k) = N
                        A = pow(g, k, p)
                        out.append(f"{A} {p}")
                        found = True
                        break
                if found:
                    break
        if not found:
            # fallback (should not happen under constraints)
            out.append("2 1")
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()