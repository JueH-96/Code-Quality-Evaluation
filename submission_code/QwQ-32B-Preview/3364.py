from typing import List

class Solution:
    def minimumValueSum(self, nums: List[int], andValues: List[int]) -> int:
        import math
        n = len(nums)
        m = len(andValues)
        
        # Precompute possible (j, i) pairs for each k
        possible_j = []
        for k in range(m):
            and_value = andValues[k]
            subarrays = []
            current_and = nums[0] if nums else 0
            j = 0
            for i in range(n):
                if i > 0:
                    current_and &= nums[i]
                while current_and < and_value and j < i:
                    current_and &= nums[j + 1]
                    j += 1
                if current_and == and_value:
                    subarrays.append((j, i + 1))
                current_and &= nums[j]
            possible_j.append(subarrays)
        
        # Initialize DP table
        dp = [[math.inf] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = 0
        
        # Fill DP table
        for k in range(1, m + 1):
            for i in range(1, n + 1):
                for j_and_i in possible_j[k - 1]:
                    j, end_i = j_and_i
                    if end_i == i:
                        dp[k][i] = min(dp[k][i], dp[k - 1][j] + nums[i - 1])
        
        # Check if a valid division exists
        if dp[m][n] == math.inf:
            return -1
        else:
            return dp[m][n]