import math
import bisect

def main():
    import sys
    input = sys.stdin.read().strip()
    N = int(input)
    if N < 2:
        print(0)
        return

    sqrt_N = math.isqrt(N)
    if sqrt_N < 2:
        primes = []
    else:
        sieve_size = sqrt_N
        sieve = [True] * (sieve_size + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(math.isqrt(sieve_size)) + 1):
            if sieve[i]:
                sieve[i*i : sieve_size+1 : i] = [False] * len(sieve[i*i : sieve_size+1 : i])
        primes = [i for i, is_p in enumerate(sieve) if is_p]

    # Compute C1: count primes p where p^8 <= N
    c1 = 0
    for p in primes:
        if p ** 8 > N:
            break
        c1 += 1

    # Compute C2: count pairs (p, q) where p < q and (p*q)^2 <= N
    c2 = 0
    for i in range(len(primes)):
        p = primes[i]
        if p > sqrt_N:
            break
        max_q = sqrt_N // p
        if max_q <= p:
            continue
        # Find the number of primes q where q > p and q <= max_q
        idx = bisect.bisect_right(primes, max_q)
        count = idx - (i + 1)
        if count > 0:
            c2 += count

    total = c1 + c2
    print(total)

if __name__ == '__main__':
    main()