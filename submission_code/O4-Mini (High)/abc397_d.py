import sys
import threading

def main():
    import sys
    import math
    import random

    sys.setrecursionlimit(10**7)
    input = sys.stdin.readline

    N = int(input().strip())

    # Miller-Rabin for n < 2^64 with deterministic bases
    def is_prime(n: int) -> bool:
        if n < 2:
            return False
        # small primes test
        small_primes = (2,3,5,7,11,13,17,19,23,29)
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

    # Pollard's Rho
    def pollards_rho(n: int) -> int:
        if n % 2 == 0:
            return 2
        if is_prime(n):
            return n
        while True:
            # random c in [1, n-1], random x in [0, n-1]
            c = random.randrange(1, n)
            x = random.randrange(0, n)
            y = x
            d = 1
            # f(x) = x^2 + c mod n
            while d == 1:
                x = (x*x + c) % n
                y = (y*y + c) % n
                y = (y*y + c) % n
                d = math.gcd(abs(x-y), n)
            if d != n:
                return d

    # factor via Pollard's Rho (non-recursive)
    def factor(n: int):
        fs = []
        stack = [n]
        while stack:
            m = stack.pop()
            if m == 1:
                continue
            if is_prime(m):
                fs.append(m)
            else:
                d = pollards_rho(m)
                # it might return m if failed, but pollards_rho retries until success
                if d == m:
                    # fallback: treat as prime
                    fs.append(m)
                else:
                    stack.append(d)
                    stack.append(m // d)
        return fs

    # Factor N
    primes = factor(N)
    # build divisor list
    from collections import Counter
    cnt = Counter(primes)
    divisors = [1]
    for p, e in cnt.items():
        new_divs = []
        # for each existing divisor, multiply by p^0, p^1, ..., p^e
        for d in divisors:
            mul = 1
            for _ in range(e+1):
                new_divs.append(d * mul)
                mul *= p
        divisors = new_divs

    # Try each a = x - y dividing N
    for a in divisors:
        b = N // a
        # We need 12*b - 3*a^2 to be a perfect square
        s2 = 12*b - 3*a*a
        if s2 <= 0:
            continue
        s = math.isqrt(s2)
        if s*s != s2:
            continue
        # y = (-3a + s) / 6 must be positive integer
        num = -3*a + s
        if num <= 0 or num % 6 != 0:
            continue
        y = num // 6
        x = y + a
        if y > 0 and x > 0:
            # verify (just to be safe)
            if x**3 - y**3 == N:
                print(x, y)
                return

    print(-1)

# Call main
if __name__ == "__main__":
    main()