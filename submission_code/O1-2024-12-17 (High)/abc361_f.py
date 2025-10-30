def main():
    import sys
    import math
    input_data = sys.stdin.read().strip()
    N = int(input_data)
    
    # Special quick check (though not strictly necessary):
    if N == 1:
        # 1 = 1^2, so there's exactly 1 such number.
        print(1)
        return
    
    # 1) Count how many perfect squares are <= N
    #    That is simply floor(sqrt(N)).
    sq_count = math.isqrt(N)
    
    # 2) We only need to generate perfect powers a^p for prime p >= 3.
    #    Reason: any number that is a perfect power with exponent >= 2
    #    either is a square (exponent=2) or has a prime divisor p of its exponent.
    #    So generating prime exponents p>=3 and taking the union with squares
    #    will give all perfect powers with exponent >=2.
    prime_exps = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]
    
    # We'll gather each p-th power list in ascending order, then merge them.
    # This avoids storing a huge hash set, which can be very large in memory.
    
    def generate_pth_powers(n, p):
        # Find integer limit ~ floor(n^(1/p)), correcting for float inaccuracies
        # We'll use exponentiation by float then fix up with a small loop.
        # (Alternatively, we could do a binary search, but the fix-up is simpler.)
        if p == 0:
            return []
        
        # Compute approximate limit as exp(log(n)/p).
        limit = int(n**(1.0/p))
        # Fix upwards if (limit+1)^p <= n
        while pow(limit+1, p) <= n:
            limit += 1
        # Fix downwards if limit^p > n
        while limit > 0 and pow(limit, p) > n:
            limit -= 1
        if limit < 1:
            return []
        
        # Generate p-th powers from 1..limit
        # For large p the limit will be small. For p=3 it can be up to 10^6.
        # We'll return a sorted list.
        if p == 3:
            # Slight speedup for cubes
            return [a*a*a for a in range(1, limit+1)]
        else:
            return [pow(a, p) for a in range(1, limit+1)]
    
    # Build lists for each prime exponent
    from heapq import merge
    
    lists = []
    for p in prime_exps:
        plst = generate_pth_powers(N, p)
        lists.append(plst)
    
    # Multi-way merge of all lists (each sorted in ascending order)
    merged_iter = merge(*lists)
    
    # De-duplicate while merging
    distinct_powers = []
    prev = -1
    for x in merged_iter:
        if x != prev:
            distinct_powers.append(x)
            prev = x
    
    # Now count how many of these merged perfect powers are also perfect squares
    # (to handle overlap with squares)
    overlap = 0
    isqrt = math.isqrt
    for val in distinct_powers:
        r = isqrt(val)
        if r*r == val:
            overlap += 1
    
    # The total count of distinct perfect powers with exponent >=2 is:
    # number_of_squares + number_of_(prime_exponent>=3) - overlap
    # because any number that is both a square and prime-exponent power
    # is counted in both, so subtract once.
    # (Note: 1 is included in both, so that is correctly handled.)
    answer = sq_count + len(distinct_powers) - overlap
    
    print(answer)

# Do not forget to call main!
if __name__ == "__main__":
    main()