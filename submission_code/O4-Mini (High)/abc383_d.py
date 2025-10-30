import sys
import math
import bisect

def main():
    data = sys.stdin.readline().split()
    if not data:
        return
    N = int(data[0])

    # M0 = floor(sqrt(N))
    M0 = math.isqrt(N)

    # Sieve primes up to M0
    limit = M0
    is_prime = bytearray(b'\x01') * (limit + 1)
    if limit >= 0:
        is_prime[0] = 0
    if limit >= 1:
        is_prime[1] = 0
    r = math.isqrt(limit)
    for i in range(2, r + 1):
        if is_prime[i]:
            step = i
            start = i * i
            # mark multiples of i as non-prime
            is_prime[start:limit+1:step] = b'\x00' * (((limit - start) // step) + 1)

    primes = [i for i in range(2, limit + 1) if is_prime[i]]

    # Count A: numbers of form p^8 <= N
    A = 0
    for p in primes:
        # p is small here, so p**8 is safe
        if p**8 <= N:
            A += 1
        else:
            break

    # Count B: numbers of form p^2 * q^2 <= N with p < q
    # which is equivalent to p*q <= sqrt(N) = M0
    B = 0
    # we only need p <= sqrt(M0), because for larger p no q>p can satisfy p*q <= M0
    L = math.isqrt(M0)
    for i, p in enumerate(primes):
        if p > L:
            break
        max_q = M0 // p
        # count how many primes q satisfy p < q <= max_q
        hi = bisect.bisect_right(primes, max_q)
        # primes[0..i] are <= p, so i+1 primes ≤ p
        B += (hi - (i + 1))

    print(A + B)

if __name__ == "__main__":
    main()