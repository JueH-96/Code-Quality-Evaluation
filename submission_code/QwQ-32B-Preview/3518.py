from typing import List

class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        n = len(b)
        
        # Initialize dp1
        dp1 = [a[0] * b[i] for i in range(n)]
        prefix_max_dp1 = [0] * n
        prefix_max_dp1[0] = dp1[0]
        for i in range(1, n):
            prefix_max_dp1[i] = max(prefix_max_dp1[i-1], dp1[i])
        
        # Initialize dp2
        dp2 = [prefix_max_dp1[i-1] + a[1] * b[i] for i in range(1, n)]
        prefix_max_dp2 = [0] * (n - 1)
        prefix_max_dp2[0] = dp2[0]
        for i in range(1, n - 1):
            prefix_max_dp2[i] = max(prefix_max_dp2[i-1], dp2[i])
        
        # Initialize dp3
        dp3 = [prefix_max_dp2[i-1] + a[2] * b[i] for i in range(2, n)]
        prefix_max_dp3 = [0] * (n - 2)
        prefix_max_dp3[0] = dp3[0]
        for i in range(1, n - 2):
            prefix_max_dp3[i] = max(prefix_max_dp3[i-1], dp3[i])
        
        # Initialize dp4
        dp4 = [prefix_max_dp3[i-1] + a[3] * b[i] for i in range(3, n)]
        
        # The answer is the maximum value in dp4
        return max(dp4)