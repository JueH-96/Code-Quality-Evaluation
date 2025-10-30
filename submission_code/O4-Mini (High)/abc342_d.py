import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    A = [int(next(it)) for _ in range(n)]
    MAX = max(A)
    # Build smallest prime factor (spf) sieve up to MAX
    spf = list(range(MAX+1))
    for i in range(2, int(MAX**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, MAX+1, i):
                if spf[j] == j:
                    spf[j] = i

    from collections import Counter
    freq = Counter()
    z = 0  # count of zeros

    # For each non-zero A[i], compute its square-free kernel f,
    # i.e. the product of primes dividing A[i] with odd exponent.
    for a in A:
        if a == 0:
            z += 1
        else:
            x = a
            f = 1
            while x > 1:
                p = spf[x]
                cnt = 0
                # count exponent mod 2
                while x % p == 0:
                    x //= p
                    cnt ^= 1
                if cnt:
                    f *= p
            freq[f] += 1

    # Count pairs among non-zero entries with identical f:
    ans = 0
    for c in freq.values():
        ans += c * (c - 1) // 2

    # Add pairs involving zeros: any pair with at least one zero is valid
    # pairs zero-zero: C(z,2), pairs zero-nonzero: z*(n-z)
    ans += z * (n - z) + z * (z - 1) // 2

    sys.stdout.write(str(ans))

if __name__ == '__main__':
    main()