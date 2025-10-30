from typing import List

class Solution:
    def lengthOfLongestSubsequence(self, nums: List[int], target: int) -> int:
        n = len(nums)
        dp = [[-1] * (target + 1) for _ in range(n + 1)]
        
        for i in range(n + 1):
            dp[i][0] = 0
        
        for i in range(1, n + 1):
            for j in range(target + 1):
                dp[i][j] = dp[i - 1][j]
                if j >= nums[i - 1] and dp[i - 1][j - nums[i - 1]] != -1:
                    dp[i][j] = max(dp[i][j], dp[i - 1][j - nums[i - 1]] + 1)
        
        return dp[n][target]

# Example usage:
# sol = Solution()
# print(sol.lengthOfLongestSubsequence([1, 2, 3, 4, 5], 9))  # Output: 3
# print(sol.lengthOfLongestSubsequence([4, 1, 3, 2, 1, 5], 7))  # Output: 4
# print(sol.lengthOfLongestSubsequence([1, 1, 5, 4, 5], 3))  # Output: -1