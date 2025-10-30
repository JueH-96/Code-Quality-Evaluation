class Solution:
    def stringCount(self, n: int) -> int:
        MOD = 10**9 + 7
        
        def power(base, exp):
            res = 1
            base %= MOD
            while exp > 0:
                if exp % 2 == 1:
                    res = (res * base) % MOD
                exp >>= 1
                base = (base * base) % MOD
            return res
            
        def inverse(n):
            return power(n, MOD - 2)
            
        fact = [1] * (n + 1)
        inv_fact = [1] * (n + 1)
        for i in range(1, n + 1):
            fact[i] = (fact[i - 1] * i) % MOD
        inv_fact[n] = inverse(fact[n])
        for i in range(n - 1, -1, -1):
            inv_fact[i] = (inv_fact[i + 1] * (i + 1)) % MOD
            
        def nCr_mod(n_val, r_val):
            if r_val < 0 or r_val > n_val:
                return 0
            num = fact[n_val]
            den = (inv_fact[r_val] * inv_fact[n_val - r_val]) % MOD
            return (num * den) % MOD
            
        def multinomial_coefficient(counts):
            n_total = sum(counts)
            coeff = fact[n_total]
            for count in counts:
                coeff = (coeff * inv_fact[count]) % MOD
            return coeff
            
        total_good_strings = 0
        for nl in range(1, n - 2):
            for ne in range(2, n - 1):
                for nt in range(1, n - 2):
                    no = n - nl - ne - nt
                    if no >= 0:
                        counts = [nl, ne, nt, no]
                        multi_coeff = multinomial_coefficient(counts)
                        term = (multi_coeff * power(23, no)) % MOD
                        total_good_strings = (total_good_strings + term) % MOD
                        
        # Corrected ranges based on deduction in thoughts
        total_count = 0
        for nl in range(1, n - 2 + 1): # 1 to n-2, inclusive
            for ne in range(2, n - 1 + 1): # 2 to n-1, inclusive
                for nt in range(1, n - 2 + 1): # 1 to n-2, inclusive
                    no = n - nl - ne - nt
                    if no >= 0:
                        counts = [nl, ne, nt, no]
                        multi_coeff = multinomial_coefficient(counts)
                        term = (multi_coeff * power(23, no)) % MOD
                        total_count = (total_count + term) % MOD

        return total_count