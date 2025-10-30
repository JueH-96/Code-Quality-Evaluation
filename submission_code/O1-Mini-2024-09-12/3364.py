from typing import List

class Solution:
    def minimumValueSum(self, nums: List[int], andValues: List[int]) -> int:
        n = len(nums)
        m = len(andValues)
        INF = float('inf')
        
        # Initialize DP array
        dp = [[INF] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = 0  # Base case: 0 subarrays with sum 0
        
        for k in range(1, m + 1):
            for i in range(1, n + 1):
                current_and = nums[i - 1]
                # Iterate backwards to find valid subarrays
                for j in range(i - 1, -1, -1):
                    current_and &= nums[j]
                    if current_and < andValues[k - 1]:
                        break  # Further left will only decrease the AND
                    if current_and == andValues[k - 1]:
                        if dp[k - 1][j] + nums[i - 1] < dp[k][i]:
                            dp[k][i] = dp[k - 1][j] + nums[i - 1]
            
            # Early termination if no possible division for current k
            if all(dp[k][i] == INF for i in range(n + 1)):
                return -1
        
        return dp[m][n] if dp[m][n] != INF else -1