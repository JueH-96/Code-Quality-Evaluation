from typing import List

class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        # Use a very small number as negative infinity
        MIN = -10**18  
        # dp[k] represents the maximum score we can achieve after choosing k elements.
        dp = [0, MIN, MIN, MIN, MIN]
        
        # Process each element in b; iterate backwards to ensure order is maintained.
        for x in b:
            for i in range(3, -1, -1):
                dp[i+1] = max(dp[i+1], dp[i] + a[i] * x)
                
        return dp[4]