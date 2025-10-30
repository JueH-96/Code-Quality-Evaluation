class Solution:
    def numberOfWays(self, n: int, x: int, y: int) -> int:
        MOD = 10**9 + 7
        
        # Number of ways to assign n performers to x stages
        ways_to_assign_stages = x ** n
        
        # Number of ways to assign scores to x stages
        ways_to_assign_scores = y ** x
        
        # Total number of ways
        total_ways = (ways_to_assign_stages * ways_to_assign_scores) % MOD
        
        return total_ways