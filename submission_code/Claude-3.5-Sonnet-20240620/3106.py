class Solution:
    def lengthOfLongestSubsequence(self, nums: List[int], target: int) -> int:
        dp = [0] + [-1] * target
        
        for num in nums:
            for t in range(target, num - 1, -1):
                if dp[t - num] != -1:
                    dp[t] = max(dp[t], dp[t - num] + 1)
        
        return dp[target]