def main():
    import math,sys

    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    N = int(input_data[0])
    
    # A number has exactly 9 divisors if and only if it is of the form:
    # 1) p^8  where p is a prime (since divisor count = 8+1=9)
    # 2) p^2 * q^2 where p and q are distinct primes (divisor count = (2+1)*(2+1)=9).
    #
    # For p^8, we must have p^8 <= N, so p <= N^(1/8)
    # For p^2 * q^2, note that p^2*q^2 = (p*q)^2 <= N  =>  p*q <= sqrt(N).
    
    answer = 0
    
    # Count numbers of the form p^8.
    N8 = N ** (1/8)
    p8_limit = int(math.floor(N8 + 1e-12))
    # Using a small Sieve for limit p8_limit (which is small, usually <=32)
    if p8_limit >= 2:
        is_prime_small = [True]*(p8_limit+1)
        is_prime_small[0] = is_prime_small[1] = False
        for i in range(2, int(p8_limit**0.5)+1):
            if is_prime_small[i]:
                for j in range(i*i, p8_limit+1, i):
                    is_prime_small[j] = False
        for p in range(2, p8_limit+1):
            if is_prime_small[p]:
                if p**8 <= N:
                    answer += 1
    
    # Count numbers of the form p^2 * q^2, which is (p*q)^2.
    # Condition: p*q <= sqrt(N)
    sqrtN = int(math.isqrt(N))  # integer square root, so p*q <= sqrt N.
    
    # We need all primes up to sqrtN. sqrtN is at most sqrt(4e12)=2e6.
    M = sqrtN
    sieve = [True]*(M+1)
    if M >= 0:
        sieve[0] = sieve[1] = False
    # Standard sieve for primes up to M.
    for i in range(2, int(M**0.5)+1):
        if sieve[i]:
            for j in range(i*i, M+1, i):
                sieve[j] = False
    primes = [i for i, isprime in enumerate(sieve) if isprime]
    
    # For each prime p, count primes q (with q>p) such that p*q <= sqrtN.
    # We use two pointers / binary search for efficiency.
    n = len(primes)
    for i in range(n):
        p = primes[i]
        # find the maximum index j such that p * primes[j] <= sqrtN.
        # We require j > i.
        lo = i+1
        hi = n - 1
        pos = i  # initialize; will update if any valid j is found.
        while lo <= hi:
            mid = (lo + hi) // 2
            if p * primes[mid] <= sqrtN:
                pos = mid
                lo = mid + 1
            else:
                hi = mid - 1
        count = pos - i
        if count > 0:
            answer += count

    sys.stdout.write(str(answer))


if __name__ == '__main__':
    main()