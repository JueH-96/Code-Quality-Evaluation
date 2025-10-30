import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    T = int(next(it))
    # Sieve primes up to 31623 for trial division
    MAXP = 31623
    sieve = [True] * (MAXP + 1)
    sieve[0] = sieve[1] = False
    primes = []
    for i in range(2, MAXP + 1):
        if sieve[i]:
            primes.append(i)
            step = i
            start = i * i
            if start <= MAXP:
                for j in range(start, MAXP + 1, step):
                    sieve[j] = False
    # small primes for quick MR filtering
    small_primes = primes[:50]
    # Bases for deterministic Miller-Rabin up to 2^64
    MR_bases = (2, 325, 9375, 28178, 450775, 9780504, 1795265022)
    
    def is_prime(n: int) -> bool:
        """Deterministic Miller-Rabin for n < 2^64"""
        if n < 2:
            return False
        # small prime check
        for p in small_primes:
            if n % p == 0:
                return n == p
        # write n-1 as d*2^s
        d = n - 1
        s = 0
        while d & 1 == 0:
            d >>= 1
            s += 1
        # test bases
        for a in MR_bases:
            if a % n == 0:
                continue
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

    # Factorization cache
    factor_cache = {}
    def factorize(n: int) -> dict:
        """Return dict of prime->exponent for n"""
        if n in factor_cache:
            return factor_cache[n]
        orig = n
        res = {}
        # trial divide by small primes
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                cnt = 0
                while n % p == 0:
                    n //= p
                    cnt += 1
                res[p] = cnt
        if n > 1:
            res[n] = res.get(n, 0) + 1
        factor_cache[orig] = res
        return res

    out = []
    for _ in range(T):
        N = int(next(it))
        if N == 1:
            # n=1: choose A=2, M=1
            out.append("2 1")
            continue
        # factor N
        facN = factorize(N)
        # search for prime P = N*k + 1
        k = 1
        while True:
            P = N * k + 1
            if is_prime(P):
                # factor k
                fack = factorize(k)
                # prime divisors of P-1 = N*k
                prime_divs = set(facN.keys()) | set(fack.keys())
                # find primitive root g mod P
                # primitive iff for all q|P-1: g^((P-1)/q) != 1 mod P
                g = 2
                # Precompute P-1
                pm1 = P - 1
                while True:
                    ok = True
                    for q in prime_divs:
                        if pow(g, pm1 // q, P) == 1:
                            ok = False
                            break
                    if ok:
                        break
                    g += 1
                # Compute A = g^k mod P; order of A will be (P-1)/k = N
                A = pow(g, k, P)
                out.append(f"{A} {P}")
                break
            k += 1

    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()