class Solution:
    def numberOfWays(self, n: int, x: int, y: int) -> int:
        MOD = 10**9 + 7
        
        # Calculate x^n % MOD using fast exponentiation
        def power(base, exp, mod):
            result = 1
            while exp > 0:
                if exp % 2 == 1:
                    result = (result * base) % mod
                base = (base * base) % mod
                exp //= 2
            return result
        
        # Calculate the number of ways to assign n performers to x stages
        stage_assignments = power(x, n, MOD)
        
        # Calculate the number of ways to score the bands
        # We need to consider all possible non-empty subsets of performers that form bands
        # and each of these can receive a score from 1 to y.
        # The number of non-empty subsets of n items is 2^n - 1 (all subsets minus the empty set)
        non_empty_subsets = (power(2, n, MOD) - 1 + MOD) % MOD
        
        # Each non-empty subset can be scored in y ways
        scoring_ways = power(y, non_empty_subsets, MOD)
        
        # Total ways is the product of stage assignments and scoring ways
        total_ways = (stage_assignments * scoring_ways) % MOD
        
        return total_ways