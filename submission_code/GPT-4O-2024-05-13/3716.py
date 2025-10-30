class Solution:
    def longestSubsequence(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        # dp[i] will store the length of the longest subsequence ending at index i
        dp = [1] * n
        
        for i in range(1, n):
            for j in range(i):
                if abs(nums[i] - nums[j]) <= abs(nums[j] - nums[j-1]) if j > 0 else True:
                    dp[i] = max(dp[i], dp[j] + 1)
        
        return max(dp)