class Solution:
    def lengthOfLongestSubsequence(self, nums: List[int], target: int) -> int:
        dp = [0] + [-10**9] * target
        
        for num in nums:
            for j in range(target, num - 1, -1):
                if dp[j - num] != -10**9:
                    dp[j] = max(dp[j], dp[j - num] + 1)
        
        return dp[target] if dp[target] != -10**9 else -1