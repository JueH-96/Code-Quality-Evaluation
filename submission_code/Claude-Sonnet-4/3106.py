class Solution:
    def lengthOfLongestSubsequence(self, nums: List[int], target: int) -> int:
        # dp[i] represents the maximum length of subsequence that sums to i
        # Initialize with -1 (impossible) except dp[0] = 0 (empty subsequence)
        dp = [-1] * (target + 1)
        dp[0] = 0
        
        for num in nums:
            # Iterate backwards to avoid using the same element multiple times
            for i in range(target, num - 1, -1):
                if dp[i - num] != -1:
                    dp[i] = max(dp[i], dp[i - num] + 1)
        
        return dp[target]