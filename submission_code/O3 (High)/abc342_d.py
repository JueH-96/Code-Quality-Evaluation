import sys

def build_spf(limit: int):
    """
    Smallest Prime Factor sieve.
    spf[x]  (x >= 2) = the smallest prime factor of x.
    """
    spf = list(range(limit + 1))
    for p in range(2, int(limit ** 0.5) + 1):
        if spf[p] == p:                   # p is a prime
            for q in range(p * p, limit + 1, p):
                if spf[q] == q:
                    spf[q] = p
    return spf


def square_free_kernel(x: int, spf, cache):
    """
    Return the square-free kernel of x:
    product of primes whose exponents are odd in the factorisation of x.
    (This uniquely represents the parity vector of exponents.)
    """
    if x in cache:
        return cache[x]

    n = x
    res = 1
    while n > 1:
        p = spf[n]
        cnt = 0
        while n % p == 0:
            n //= p
            cnt ^= 1                 # we only need the parity
        if cnt:                      # odd multiplicity ⇒ keep the prime once
            res *= p
    cache[x] = res
    return res


def main() -> None:
    data = list(map(int, sys.stdin.buffer.read().split()))
    N, A = data[0], data[1:]

    # number of zeros ---------------------------------------------------------
    c0 = A.count(0)

    # -------------------------------------------------------------------------
    # prepare SPF up to the maximal positive value that occurs
    # (if there are no positive numbers, limit 1 is enough)
    # -------------------------------------------------------------------------
    positives = [x for x in A if x]
    limit = max(positives) if positives else 1
    spf = build_spf(limit)

    # -------------------------------------------------------------------------
    # count how many numbers share the same square-free kernel
    # -------------------------------------------------------------------------
    cache = {}                 # value  -> kernel (memoisation)
    groups = {}                # kernel -> frequency
    for v in positives:
        k = square_free_kernel(v, spf, cache)
        groups[k] = groups.get(k, 0) + 1

    # -------------------------------------------------------------------------
    # pairs whose product is a perfect square
    # 1) pairs that contain at least one zero
    # 2) pairs of positive numbers that have identical kernels
    # -------------------------------------------------------------------------
    ans = 0

    # 1) involving zeros
    ans += c0 * (N - c0)               # exactly one zero
    ans += c0 * (c0 - 1) // 2          # both zeros

    # 2) positive numbers with identical kernels
    for freq in groups.values():
        ans += freq * (freq - 1) // 2

    print(ans)


if __name__ == "__main__":
    main()