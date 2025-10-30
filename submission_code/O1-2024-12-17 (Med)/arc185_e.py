def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))

    MOD = 998244353
    MAXA = 10**5

    # Precompute powers of 2 up to N (so we can do 2^(m-1) mod M quickly)
    pow2 = [1]*(N+1)
    for i in range(1, N+1):
        pow2[i] = (pow2[i-1] << 1) % MOD

    # We will need:
    #  1) A list of all divisors for each 1 <= x <= MAXA
    #  2) The distinct prime factors of each x (for fast inclusion-exclusion)

    # Step 1: Sieve of smallest prime to help factor quickly
    spf = [0]*(MAXA+1)  # smallest prime factor
    spf[1] = 1
    for i in range(2, MAXA+1):
        if spf[i] == 0:
            # i is prime
            spf[i] = i
            if i*i <= MAXA:
                for j in range(i*i, MAXA+1, i):
                    if spf[j] == 0:
                        spf[j] = i

    # Step 2: Precompute distinct prime factors for each x
    # and also precompute the full list of divisors for each x
    # To save time and memory, we do it on the fly and cache results.
    # (Precomputing for all up to 100000 is also possible; here we'll do an on-demand cache.)
    # But typically we do want them all ready, because we'll need them many times.

    # However, a full precomputation for all x up to 100000 of divisors and prime factors
    # is still fairly standard and should be acceptable in memory (~ up to a few million ints).
    # We'll implement a standard method: for each x, factor it using spf, then generate divisors.

    # We'll build two global arrays:
    #    divisors_list[x] = sorted list of all divisors of x
    #    primefactors_list[x] = list of distinct prime factors of x
    # Because we need them repeatedly for each A[m].

    divisors_list = [[] for _ in range(MAXA+1)]
    primefactors_list = [[] for _ in range(MAXA+1)]

    # A helper function to factor out distinct primes of x using spf:
    def get_distinct_primes(x):
        pf = []
        lastp = 0
        while x > 1:
            p = spf[x]
            if p != lastp:
                pf.append(p)
                lastp = p
            x //= p
        return pf

    # We'll fill in primefactors_list[i] for all i
    for i in range(1, MAXA+1):
        primefactors_list[i] = get_distinct_primes(i)

    # Next we precompute divisors for all i using a classic approach:
    for i in range(1, MAXA+1):
        for j in range(i, MAXA+1, i):
            divisors_list[j].append(i)

    # c'[r] = sum over all k multiple of r of freq[k].
    # But we never actually need freq[k] alone; we only need to do partial updates:
    #   - When we add +val to freq[x], we also add +val to c'[d] for each d dividing x.
    # Then c'[r] can be queried in O(1).
    c_ = [0]*(MAXA+1)  # c'[i] array
    freq = [0]*(MAXA+1)  # freq[x] = sum of 2^(j-1) for all j with A_j = x, mod M

    # f(m) will be our running "sum of scores of all subsequences up to m"
    f_prev = 0  # f(0)=0

    # A helper function to compute G(m) = sum_{x} gcd(x,n)*freq[x],
    # using the well-known decomposition:
    #    sum_{x} gcd(x,n)*freq[x]
    #  = sum_{d|n} d * ( sum_{x : gcd(x,n) = d} freq[x] ).
    #  Let n1 = n/d. Then gcd(x,n)=d <=> x is multiple of d and gcd(x/d, n1)=1.
    #  So sum_{x : gcd(x,n)=d} freq[x] = sum_{t : gcd(t,n1)=1} freq[d*t].
    #
    # We'll do inclusion-exclusion on the prime factors of n1 to compute S(d) = sum_{t gcd(t,n1)=1} freq[d*t].
    # That means:
    #    S(d, n1) = sum_{t gcd(t,n1)=1} freq[d*t]
    #              = sum_{subset of prime_factors(n1)} ((-1)^|subset|) * c'[d*(product_of_subset_pf)].
    #
    # We'll implement a DFS over subsets of prime_factors(n1).

    def sum_gcd_freq(n):
        # sum_{x} gcd(x,n)*freq[x]
        # We'll iterate over divisors d of n
        #   n1 = n//d
        #   S(d, n1) = inclusion-exclusion on prime factors of n1
        #   add up d * S(d, n1)
        total = 0
        divs = divisors_list[n]
        for d in divs:
            n1 = n // d
            # gather prime factors of n1
            pf = primefactors_list[n1]
            # inclusion-exclusion
            s = 0

            # We'll do a small DFS or iterative subsets:
            # Careful with performance: pf can have up to ~6-7 distinct primes in worst case
            # We'll define a stack-based or recursive approach:

            def rec(idx, cur_mul, sign):
                nonlocal s
                if idx == len(pf):
                    s = (s + sign * c_[d*cur_mul]) % MOD
                    return
                # option 1: skip pf[idx]
                rec(idx+1, cur_mul, sign)
                # option 2: include pf[idx]
                mul2 = cur_mul * pf[idx]
                if d*mul2 <= MAXA:
                    rec(idx+1, mul2, -sign)

            # call rec
            rec(0, 1, 1)

            total = (total + d*s) % MOD

        return total

    # We will store the answers in a list and print at the end
    out = []
    # Process each m = 1..N
    # f(m) = 2*f(m-1) + G(m), where G(m) = sum_{j=1..m-1} gcd(A_j, A_m)*2^(j-1).
    # But effectively G(m) = sum_{x} gcd(x, A_m)* freq[x], if freq[x] = sum_{j: A_j=x} 2^(j-1).
    for m in range(1, N+1):
        x = A[m-1]
        # Compute the extra term G(m):
        # G(m) = sum_{k} gcd(k,x) * freq[k] = sum_gcd_freq(x)
        add_val = sum_gcd_freq(x)
        f_m = (2*f_prev + add_val) % MOD

        # Output f(m)
        out.append(str(f_m))

        # Now update freq[x] by 2^(m-1) (the new single-element subsequence chooses A_m)
        inc_val = pow2[m-1]
        freq[x] = (freq[x] + inc_val) % MOD

        # Also update c'[d] for each divisor d of x
        for d in divisors_list[x]:
            c_[d] = (c_[d] + inc_val) % MOD

        f_prev = f_m

    print("
".join(out))