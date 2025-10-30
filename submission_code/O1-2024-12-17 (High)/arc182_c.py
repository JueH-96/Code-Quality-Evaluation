def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    M = int(input_data[1])
    MOD = 998244353

    # 1) Gather all primes up to M (M <= 16).
    #    We only need primes up to M because M <= 16.
    def is_prime(x):
        if x < 2:
            return False
        for v in range(2, int(x**0.5) + 1):
            if x % v == 0:
                return False
        return True

    primes = []
    for x in range(2, M+1):
        if is_prime(x):
            primes.append(x)
    r = len(primes)  # number of distinct primes up to M

    # 2) For each i in [1..M], factor it wrt these primes and record exponents.
    #    e[i][p] = exponent of prime p in i
    #    We'll store e(i) as a list of length r.
    #    Note: use 0-based index for i, i.e. i from 0..M-1 represents the integer (i+1).
    e = [[0]*r for _ in range(M)]
    for i in range(M):
        val = i+1
        for p_idx, p in enumerate(primes):
            cnt = 0
            while val % p == 0:
                val //= p
                cnt += 1
            e[i][p_idx] = cnt

    # 3) Build the vector F of size 2^r:
    #    For each subset S of {0..r-1}, F[S] = sum_{i=1..M} ( product of e_p(i) for p in S ).
    #    Equivalently, if S = {p1, ..., pk}, F[S] = sum_{i=1..M} [ e[p1](i) * ... * e[pk](i) ].
    #    Also note F[∅] = sum_{i=1..M} 1 = M.
    size = 1 << r
    F = [0]*size
    for mask in range(size):
        # For each i in [1..M], compute product of exponents e_p(i) for p in mask.
        ssum = 0
        for i in range(M):
            prod = 1
            # go through bits of mask
            # if (mask & (1<<p_idx)) != 0 => multiply by e[i][p_idx]
            tmp_val = 1
            mbit = mask
            p_idx = 0
            while mbit:
                # isolate next set bit
                # we can do a trick: lowbit = (mbit & -mbit).
                # but simpler: just check each prime
                # however, we can do a while version:
                bit = (mbit & -mbit)
                bit_index = (bit).bit_length() - 1
                tmp_val *= e[i][bit_index]
                mbit ^= bit
            ssum = (ssum + tmp_val) % MOD
        F[mask] = ssum

    # The "identity" E for subset convolution (size 2^r) is:
    # E[∅] = 1, E[S] = 0 for S != ∅.
    E = [0]*size
    E[0] = 1

    # 4) Define subset-convolution in O(3^r), r <= 6 => at most 729 ops, which is fast enough.
    def subset_conv(A, B):
        C = [0]*size
        for s in range(size):
            # enumerate subsets t of s
            t = s
            ssum = 0
            while True:
                ssum += A[t]*B[s^t]
                if ssum >= 1<<61:  # to prevent Python int growing too large, mod occasionally
                    ssum %= MOD
                if t == 0:
                    break
                t = (t-1) & s
            C[s] = ssum % MOD
        return C

    # Vector add/sub mod
    def vec_add(A, B):
        return [( (A[i] + B[i]) % MOD ) for i in range(size)]
    def vec_sub(A, B):
        return [( (A[i] - B[i]) % MOD ) for i in range(size)]

    # 5) We want to compute (F^N, sum_{k=0..N} F^k) in the "ring" of size-2^r vectors
    #    under the subset convolution "*" defined above.
    #    Then our final answer = sum_{S} ( sum_{k=0..N}F^k[S] ) - 1 (mod) if N>=1.
    #    Because sum_{k=1..N} = sum_{k=0..N} - F^0, and F^0 is the identity E => sum(E)=1.
    # We'll implement a fast-exponent-with-sum "powSum(F,n)" via divide-and-conquer.

    def pow_sum(F, n):
        """
        Returns (X, Y) where:
          X = F^n  (subset-convolution exponent)
          Y = sum_{k=0..n} F^k
        in mod 998244353.
        """
        if n == 0:
            # F^0 = E, sum_{k=0..0} F^k = E
            return (E, E)

        # recursively compute half
        (A, B) = pow_sum(F, n >> 1)  # A=F^(n//2), B= ∑_{k=0..n//2} F^k
        A2 = subset_conv(A, A)       # A2= F^(2*(n//2)) => F^(n - (n%2))
        # B2= sum_{k=0..2*(n//2)} F^k => B + A*(B - E)
        # i.e. S(2m)= S(m) + F^m (S(m) - I)
        tmp = vec_sub(B, E)         # (B-E)
        ABmE = subset_conv(A, tmp)  # A*(B - E)
        B2 = vec_add(B, ABmE)       # S(2m) if n even, or S(2m+ something)

        if (n & 1) == 0:
            # even => n = 2*(n//2)
            return (A2, B2)
        else:
            # odd => n = 2*(n//2) + 1
            # F^n = A2 * F
            A2F = subset_conv(A2, F)
            # sum_{k=0..n} F^k = sum_{k=0..2m} + F^(2m+1) => B2 + A2F
            B2F = vec_add(B2, A2F)
            return (A2F, B2F)

    # 6) Compute (F^N, sum_{k=0..N} F^k) then get the final answer:
    #    If N=0, sum_{k=1..N} is empty => 0.  But per the problem constraints N >=1 anyway.
    #    For N>=1 => final = sum_{S}( sum_{k=0..N}[F^k[S]] ) - 1 = sum_{S}( Y[S] ) - 1.
    if N == 0:
        # By the problem statement, N >= 1. 
        # But in case it ever were 0, the sum_{k=1..0} = 0
        print(0)
        return

    (FN, SUMF) = pow_sum(F, N)  # SUMF = sum_{k=0..N} F^k

    ans = 0
    for val in SUMF:
        ans = (ans + val) % MOD
    # subtract 1 for the identity term F^0
    ans = (ans - 1) % MOD

    print(ans)

# Do not forget to call main()
main()