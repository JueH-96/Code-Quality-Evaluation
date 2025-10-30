class Solution:
    def stringCount(self, n: int) -> int:
        if n < 4:
            return 0
            
        MOD = 10**9 + 7
        
        # Total possible strings
        total = pow(26, n, MOD)
        
        # Count strings missing required letters using inclusion-exclusion
        # We need: at least 1 'l', at least 2 'e's, at least 1 't'
        
        # Single sets
        no_l = pow(25, n, MOD)
        at_most_1_e = (pow(25, n, MOD) + n * pow(25, n-1, MOD)) % MOD
        no_t = pow(25, n, MOD)
        
        # Pairwise intersections
        no_l_and_at_most_1_e = (pow(24, n, MOD) + n * pow(24, n-1, MOD)) % MOD
        no_l_and_no_t = pow(24, n, MOD)
        no_t_and_at_most_1_e = (pow(24, n, MOD) + n * pow(24, n-1, MOD)) % MOD
        
        # Triple intersection
        all_three = (pow(23, n, MOD) + n * pow(23, n-1, MOD)) % MOD
        
        # Apply inclusion-exclusion
        excluded = (no_l + at_most_1_e + no_t - no_l_and_at_most_1_e 
                   - no_l_and_no_t - no_t_and_at_most_1_e + all_three) % MOD
        
        return (total - excluded + MOD) % MOD