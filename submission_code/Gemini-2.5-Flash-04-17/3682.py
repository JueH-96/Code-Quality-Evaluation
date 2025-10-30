class Solution:
    def countGoodArrays(self, n: int, m: int, k: int) -> int:
        MOD = 10**9 + 7
        
        # Handle the special case m=1
        # If m=1, all elements must be 1. The array is [1, 1, ..., 1].
        # In this array, all n-1 adjacent pairs are equal.
        # So, a good array exists only if k == n-1. If it exists, there is only 1 such array.
        if m == 1:
            return 1 if k == n - 1 else 0
        
        # Function for modular exponentiation (base^exp % mod)
        def power(base, exp, mod):
            res = 1
            base %= mod
            while exp > 0:
                if exp % 2 == 1:
                    res = (res * base) % mod
                base = (base * base) % mod
                exp //= 2
            return res

        # We need combinations C(N, K) where N=n-1 and K=k.
        # We need factorials and inverse factorials up to max(N, K, N-K) = max(n-1, k, n-1-k) = n-1.
        # We need arrays of size n to store values from index 0 to n-1.
        
        fact = [0] * n
        invFact = [0] * n
        
        # Precompute factorials modulo MOD
        fact[0] = 1
        for i in range(1, n):
            fact[i] = (fact[i-1] * i) % MOD

        # Precompute inverse factorials modulo MOD using Fermat's Little Theorem
        # invFact[i] = (i!)^(-1) mod MOD
        # invFact[n-1] = (fact[n-1])^(MOD-2) mod MOD
        # invFact[i] = invFact[i+1] * (i+1) mod MOD for i < n-1
        
        # Constraints guarantee n >= 1.
        # fact[n-1] is needed, index is n-1. Array size n is sufficient.
        # invFact[n-1] needs fact[n-1].
        invFact[n-1] = power(fact[n-1], MOD - 2, MOD)
        
        # If n=1, range(1-2, -1, -1) is range(-1, -1, -1), which is empty.
        # If n > 1, the loop runs from n-2 down to 0.
        # This correctly computes invFact[i] using invFact[i+1].
        # invFact[0] needs invFact[1], ..., invFact[n-2] needs invFact[n-1].
        # The loop order is correct.
        for i in range(n - 2, -1, -1):
            invFact[i] = (invFact[i+1] * (i+1)) % MOD
        
        # Function for combinations C(N, K) modulo MOD
        def combinations(N, K, fact, invFact, mod):
            if K < 0 or K > N:
                return 0
            # C(N, K) = N! / (K! * (N-K)!) mod P
            # = N! * (K!)^-1 * ((N-K)!)^-1 mod P
            numerator = fact[N]
            denominator = (invFact[K] * invFact[N-K]) % mod
            return (numerator * denominator) % mod

        # We need to choose k positions out of n-1 positions for equality
        # N = n-1, K = k
        # Constraints: 0 <= k <= n-1. This means N >= 0, K >= 0, N-K >= 0.
        # All indices N, K, N-K are within the range [0, n-1].
        # fact and invFact arrays are indexed up to n-1, which covers these indices.
        C_n_minus_1_k = combinations(n - 1, k, fact, invFact, MOD)
        
        # Number of ways to assign values to the segments.
        # There are n-k segments of consecutive equal elements.
        # The positions where adjacent elements are unequal are the boundaries between segments.
        # There are (n-1) - k unequal positions, which divide the array into (n-1 - k) + 1 = n-k segments.
        # The first segment has m choices (any value from 1 to m).
        # The remaining n-k-1 segments must have values different from the previous segment.
        # For each of the subsequent segments, there are (m-1) choices.
        # This logic applies when n-k >= 1, which is true because k <= n-1.
        # The total ways to color the segments is m * (m-1)^(n-k-1).
        
        term_m = m % MOD
        
        # The exponent for (m-1) is (n-k-1).
        # Since k <= n-1, n-k >= 1, so n-k-1 >= 0. The exponent is non-negative.
        power_base = m - 1
        power_exponent = n - k - 1
        
        # If m=1, power_base = 0. This case is handled separately at the top,
        # so m > 1 when we reach this point. Hence, power_base >= 1.
        # The power function handles base >= 1 and exponent >= 0 correctly.
        
        term_power = power(power_base, power_exponent, MOD)
        
        # Total count = C(n-1, k) * m * (m-1)^(n-k-1) mod MOD
        
        result = (C_n_minus_1_k * term_m) % MOD
        result = (result * term_power) % MOD
        
        return result