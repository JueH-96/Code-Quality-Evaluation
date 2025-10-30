class Solution:
    def lengthOfLongestSubsequence(self, nums: List[int], target: int) -> int:
        dp = [-float('inf')] * (target + 1)
        dp[0] = 0  # Base case: 0 sum with 0 elements
        
        for num in nums:
            for j in range(target, num - 1, -1):
                if dp[j - num] + 1 > dp[j]:
                    dp[j] = dp[j - num] + 1
        
        return dp[target] if dp[target] != -float('inf') else -1