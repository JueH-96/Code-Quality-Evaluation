class Solution:
    def numberOfWays(self, n: int, x: int, y: int) -> int:
        MOD = 10**9 + 7
        
        # First, calculate the number of ways to distribute n performers into x stages
        # This is x^n as each performer has x choices
        stage_distributions = pow(x, n, MOD)
        
        # Now calculate how many non-empty stages we can have
        # For each stage distribution, we need to consider how many stages are actually used
        # We'll use the principle of inclusion-exclusion
        
        # For each possible number of non-empty stages (from 1 to min(n,x))
        total = 0
        for k in range(1, min(n + 1, x + 1)):
            # Calculate C(x,k) * k^n * y^k
            # C(x,k) - ways to choose k stages from x stages
            # k^n - ways to distribute n performers to exactly k stages
            # y^k - ways to assign scores to k bands
            
            # Calculate C(x,k)
            c = 1
            for i in range(k):
                c = c * (x - i) // (i + 1)
            
            # Calculate k^n
            stage_ways = pow(k, n, MOD)
            
            # Calculate y^k
            score_ways = pow(y, k, MOD)
            
            # Multiply all together
            term = (c * stage_ways) % MOD
            term = (term * score_ways) % MOD
            
            # Apply inclusion-exclusion principle
            if k % 2 == 1:
                total = (total + term) % MOD
            else:
                total = (total - term) % MOD
        
        return total