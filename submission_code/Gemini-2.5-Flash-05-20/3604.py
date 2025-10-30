class Solution:
    def numberOfWays(self, n: int, x: int, y: int) -> int:
        MOD = 10**9 + 7
        MAX_N_X = max(n, x)

        # Precompute factorials and inverse factorials
        fact = [1] * (MAX_N_X + 1)
        invFact = [1] * (MAX_N_X + 1)

        # Helper for modular exponentiation: (base^exp) % MOD
        def power(base, exp):
            res = 1
            base %= MOD
            while exp > 0:
                if exp % 2 == 1:
                    res = (res * base) % MOD
                base = (base * base) % MOD
                exp //= 2
            return res

        # Calculate factorials: fact[i] = i! % MOD
        for i in range(2, MAX_N_X + 1):
            fact[i] = (fact[i-1] * i) % MOD
        
        # Calculate inverse factorials: invFact[i] = (i!)^(-1) % MOD
        # Using Fermat's Little Theorem: a^(MOD-2) % MOD for a^(-1) % MOD
        invFact[MAX_N_X] = power(fact[MAX_N_X], MOD - 2)
        # Calculate remaining inverse factorials using invFact[i] = invFact[i+1] * (i+1) % MOD
        for i in range(MAX_N_X - 1, -1, -1):
            invFact[i] = (invFact[i+1] * (i+1)) % MOD

        # Helper for nCr % MOD: (n! * (r!)^-1 * ((n-r)!)^-1) % MOD
        def nCr_mod_p(n_val, r_val):
            if r_val < 0 or r_val > n_val:
                return 0
            numerator = fact[n_val]
            denominator = (invFact[r_val] * invFact[n_val - r_val]) % MOD
            return (numerator * denominator) % MOD

        total_ways = 0

        # Iterate over k, the number of non-empty stages
        # k can range from 1 to min(n, x)
        for k in range(1, min(n, x) + 1):
            # Part 1: Number of ways to choose k stages out of x stages
            # C(x, k) ways
            ways_to_choose_k_stages = nCr_mod_p(x, k)

            # Part 2: Number of ways to assign n performers to these k specific stages
            # such that all k stages are non-empty. This is P(n, k).
            # P(n, k) = k! * S(n, k) = sum_{j=0 to k} [(-1)^(k-j) * C(k, j) * j^n]
            # (where S(n, k) is the Stirling number of the second kind)
            P_nk = 0
            for j in range(k + 1):
                term_C_kj = nCr_mod_p(k, j) # C(k, j)

                # j^n: 0^n is 0 for n > 0. Since n >= 1 (as per constraints), 
                # power(0, n) will correctly return 0.
                j_pow_n = power(j, n) 
                
                current_term_val = (term_C_kj * j_pow_n) % MOD

                # Apply (-1)^(k-j) factor
                if (k - j) % 2 == 1:  # If (k-j) is odd, it's -1
                    P_nk = (P_nk - current_term_val + MOD) % MOD
                else:  # If (k-j) is even, it's +1
                    P_nk = (P_nk + current_term_val) % MOD
            
            # Part 3: Number of ways to award scores to these k bands
            # Each band (non-empty stage) can get a score from 1 to y.
            # So, y^k ways to assign scores.
            y_pow_k = power(y, k)

            # Combine the parts for this 'k':
            # (Ways to choose stages) * (Ways to assign performers to those stages) * (Ways to score bands)
            term_for_k = (ways_to_choose_k_stages * P_nk) % MOD
            term_for_k = (term_for_k * y_pow_k) % MOD
            
            # Add to total ways
            total_ways = (total_ways + term_for_k) % MOD
            
        return total_ways