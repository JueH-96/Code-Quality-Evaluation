from typing import List

class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [float('-inf')] * n
        
        for i in range(n):
            dp[i] = nums[i]
            for j in range(i):
                if nums[i] - nums[j] >= i - j:
                    dp[i] = max(dp[i], dp[j] + nums[i])
        
        return max(dp)

# Example usage:
# sol = Solution()
# print(sol.maxBalancedSubsequenceSum([3, 3, 5, 6]))  # Output: 14
# print(sol.maxBalancedSubsequenceSum([5, -1, -3, 8]))  # Output: 13
# print(sol.maxBalancedSubsequenceSum([-2, -1]))  # Output: -1