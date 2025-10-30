class Solution:
    def numberOfWays(self, n: int, x: int, y: int) -> int:
        MOD = 10**9 + 7
        
        # Calculate the number of ways to assign stages
        # Each performer can choose one of the x stages
        stage_assignments = pow(x, n, MOD)
        
        # Calculate the number of ways to assign scores
        # Each unique band can receive a score from 1 to y
        # The number of unique bands can be at most n (if all performers are on different stages)
        score_assignments = pow(y, n, MOD)
        
        # The total number of ways is the product of stage assignments and score assignments
        total_ways = (stage_assignments * score_assignments) % MOD
        
        return total_ways