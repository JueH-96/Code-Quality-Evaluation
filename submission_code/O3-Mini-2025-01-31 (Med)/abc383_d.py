def main():
    import sys, math, bisect
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    
    # A number has exactly 9 divisors if and only if it is of the form:
    #   1) p^8, where p is prime (since 8+1 = 9 divisors)
    #   2) p^2 * q^2, where p and q are distinct primes
    #
    # For p^8: we need p^8 <= N  =>  p <= N^(1/8)
    # For p^2 * q^2: note that (p*q)^2 <= N  =>  p*q <= sqrt(N)
    
    # We'll need a list of primes up to sqrt(N) (because for the second case,
    # we need p and q such that p*q <= sqrt(N)), and note sqrt(N) is at most 2e6
    max_prime = math.isqrt(N)
    if max_prime < 2:
        sys.stdout.write("0")
        return

    # Sieve of Eratosthenes to generate primes up to max_prime
    sieve = bytearray(b'\x01') * (max_prime + 1)
    sieve[:2] = b'\x00\x00'
    for i in range(2, int(max_prime ** 0.5) + 1):
        if sieve[i]:
            start = i * i
            sieve[start:max_prime + 1:i] = b'\x00' * (((max_prime - start) // i) + 1)
    primes = [i for i, isprime in enumerate(sieve) if isprime]
    
    total_count = 0
    # Count numbers of the form p^8
    # p must be <= N^(1/8)
    limit8 = int(N ** (1/8))
    for p in primes:
        if p > limit8:
            break
        if p ** 8 <= N:
            total_count += 1
        else:
            break

    # Count numbers of the form p^2 * q^2 (equivalently, (p*q)^2 <= N)
    # This is equivalent to p*q <= sqrt(N)
    sqrtN = math.isqrt(N)
    # For each prime p, count primes q (with q > p) such that p*q <= sqrtN.
    for i, p in enumerate(primes):
        if p > sqrtN:
            break
        max_q = sqrtN // p
        # Using binary search to find the number of primes <= max_q:
        j = bisect.bisect_right(primes, max_q)
        # We need only primes with index greater than i (i.e. q > p)
        count_for_p = j - (i + 1)
        if count_for_p > 0:
            total_count += count_for_p

    sys.stdout.write(str(total_count))

if __name__ == '__main__':
    main()