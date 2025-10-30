class Solution:
    def lengthOfLongestSubsequence(self, nums: List[int], target: int) -> int:
        n = len(nums)
        dp = [[0] * (target + 1) for _ in range(n + 1)]
        
        for i in range(1, n + 1):
            for j in range(1, target + 1):
                if nums[i - 1] <= j:
                    dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - nums[i - 1]] + 1)
                else:
                    dp[i][j] = dp[i - 1][j]
        
        for j in range(target, -1, -1):
            if dp[n][j] > 0:
                return dp[n][j]
        
        return -1