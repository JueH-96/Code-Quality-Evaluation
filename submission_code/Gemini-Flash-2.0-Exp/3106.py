class Solution:
    def lengthOfLongestSubsequence(self, nums: List[int], target: int) -> int:
        n = len(nums)
        dp = [[-float('inf')] * (target + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        
        for i in range(1, n + 1):
            for j in range(target + 1):
                dp[i][j] = dp[i-1][j]
                if j >= nums[i-1]:
                    dp[i][j] = max(dp[i][j], dp[i-1][j - nums[i-1]] + 1)
        
        if dp[n][target] == -float('inf'):
            return -1
        else:
            return dp[n][target]