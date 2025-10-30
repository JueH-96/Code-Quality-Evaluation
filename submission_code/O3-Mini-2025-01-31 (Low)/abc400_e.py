def main():
    import sys, math, bisect
    
    data = sys.stdin.read().split()
    if not data:
        return
    Q = int(data[0])
    queries = list(map(int, data[1:]))
    
    MAXN = 10**12
    
    # Sieve for all primes up to 10^6.
    # Why 10^6? In a 400 number N = p^(2a)*q^(2b) with a,b>=1, we have p^(2a) * q^(2b) ≤ 10^12.
    # In the smallest exponent case (a=b=1) we get N = p^2*q^2 ≤ 10^12 so that p*q ≤ 10^6.
    max_limit = 10**6
    sieve = bytearray(b'\x01') * (max_limit + 1)
    sieve[:2] = b'\x00\x00'
    for i in range(2, int(max_limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i:max_limit+1:i] = b'\x00' * len(range(i*i, max_limit+1, i))
    primes = [i for i, isprime in enumerate(sieve) if isprime]
    
    # A "400 number" N must have exactly 2 distinct prime factors; and each appears with an even exponent.
    # So we can write N = p^(2a) * q^(2b) with distinct primes p < q and a,b >= 1.
    # We now precompute all such N <= 10^12.
    #
    # Note: We iterate over possible exponent pairs (a,b).
    # We only consider (a,b) for which the smallest possible candidate using p = 2 and q = 3 is <= MAXN.
    # That is, 2^(2a) * 3^(2b) <= MAXN.
    valid_exp = []
    max_exp = 40  # safe upper bound for exponents
    for a in range(1, max_exp):
        if 2**(2*a) > MAXN:
            break
        for b in range(1, max_exp):
            if 2**(2*a) * 3**(2*b) > MAXN:
                break
            valid_exp.append((a, b))
    
    result_nums = []
    # For each valid exponent pair (a, b) generate numbers:
    #   p^(2a) * q^(2b) with p and q primes, p < q,
    # and with the restrictions that the number is <= MAXN.
    for a, b in valid_exp:
        # For fixed a, the factor p^(2a) must be <= MAXN. Thus p <= MAXN^(1/(2a))
        p_bound = int(pow(MAXN, 1 / (2 * a)))
        # Find the index in the primes list where primes exceed p_bound.
        idx_bound = bisect.bisect_right(primes, p_bound)
        for i in range(idx_bound):
            p = primes[i]
            p_power = p**(2 * a)
            if p_power > MAXN:
                break
            # Now, for a given p, choose q.
            # q^(2b) must be <= MAXN / p_power
            # so q <= (MAXN / p_power)^(1/(2b))
            q_bound = int(pow(MAXN / p_power, 1 / (2 * b)))
            # q must be a prime strictly greater than p.
            j = bisect.bisect_right(primes, p)
            j_upper = bisect.bisect_right(primes, q_bound)
            for q in primes[j:j_upper]:
                val = p_power * (q**(2*b))
                if val <= MAXN:
                    result_nums.append(val)
    
    # Remove duplicates (if any) and sort the list.
    result_nums = sorted(set(result_nums))
    
    # For each query A, we need the largest 400 number not exceeding A.
    # Use binary search on the sorted list.
    out_lines = []
    for A in queries:
        pos = bisect.bisect_right(result_nums, A)
        # pos is the index where A would be inserted so that all elements before pos are ≤ A.
        out_lines.append(str(result_nums[pos - 1]))
    
    sys.stdout.write("
".join(out_lines))
    
if __name__ == '__main__':
    main()