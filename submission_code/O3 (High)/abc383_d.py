import sys
import math
import bisect

def sieve(limit: int):
    """
    Classic Eratosthenes sieve.
    Returns a list with all primes in the closed interval [2, limit].
    """
    if limit < 2:
        return []
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0:2] = b'\x00\x00'
    upper = int(limit ** 0.5) + 1
    for p in range(2, upper):
        if is_prime[p]:
            is_prime[p * p: limit + 1: p] = b'\x00' * ((limit - p * p) // p + 1)
    return [i for i in range(2, limit + 1) if is_prime[i]]


def kth_integer_root(n: int, k: int) -> int:
    """
    Largest integer r such that r**k <= n   (  n >= 1, k >= 1  )
    Uses binary search; k is tiny (k=8 here) so speed is not an issue.
    """
    lo, hi = 1, int(round(n ** (1.0 / k))) + 2  # small upper overshoot
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if pow(mid, k) <= n:
            lo = mid
        else:
            hi = mid - 1
    return lo


def count_two_prime_squares(primes, m):
    """
    Count unordered pairs (p, q) of distinct primes (p<q) such that p*q <= m.
    primes must be a sorted list of all primes up to m.
    Two-pointer method works in O(π(m)).
    """
    total = 0
    j = len(primes) - 1
    for i, p in enumerate(primes):
        while j > i and p * primes[j] > m:
            j -= 1
        if j <= i:        # no more pairs will be possible
            break
        total += (j - i)  # primes[i+1 .. j] all make a valid pair with p
    return total


def main() -> None:
    n = int(sys.stdin.readline().strip())

    # largest possible m = floor(sqrt(n))    (since (p*q)^2 ≤ n  ⇔ p*q ≤ sqrt(n))
    m = math.isqrt(n)

    # Sieve all primes up to m (m ≤ 2,000,000 when n ≤ 4·10¹²)
    primes = sieve(m)

    # 1) numbers of the form p^8
    p8_limit = kth_integer_root(n, 8)
    cnt_p8 = bisect.bisect_right(primes, p8_limit)

    # 2) numbers of the form p^2 * q^2   with p < q both prime
    cnt_pairs = count_two_prime_squares(primes, m)

    print(cnt_p8 + cnt_pairs)


if __name__ == "__main__":
    main()