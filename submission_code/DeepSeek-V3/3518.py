from typing import List

class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        n = len(b)
        # Precompute the possible values for each position
        # We need to choose 4 indices i0 < i1 < i2 < i3
        # The score is a[0]*b[i0] + a[1]*b[i1] + a[2]*b[i2] + a[3]*b[i3]
        
        # To maximize the score, we need to find the best combination of b[i0], b[i1], b[i2], b[i3]
        # such that i0 < i1 < i2 < i3.
        
        # We can use dynamic programming to keep track of the best possible values at each step.
        
        # Initialize DP arrays
        # dp0[i] will store the best value for the first element up to index i
        # dp1[i] will store the best value for the first two elements up to index i
        # dp2[i] will store the best value for the first three elements up to index i
        # dp3[i] will store the best value for all four elements up to index i
        
        dp0 = [float('-inf')] * n
        dp1 = [float('-inf')] * n
        dp2 = [float('-inf')] * n
        dp3 = [float('-inf')] * n
        
        # Initialize dp0
        for i in range(n):
            dp0[i] = a[0] * b[i]
        
        # Compute dp1
        for i in range(1, n):
            dp1[i] = max(dp1[i-1], dp0[i-1] + a[1] * b[i])
        
        # Compute dp2
        for i in range(2, n):
            dp2[i] = max(dp2[i-1], dp1[i-1] + a[2] * b[i])
        
        # Compute dp3
        for i in range(3, n):
            dp3[i] = max(dp3[i-1], dp2[i-1] + a[3] * b[i])
        
        # The maximum score is the maximum value in dp3
        return max(dp3)