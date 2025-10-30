class Solution:
    def findMaximumLength(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return 1
        
        # dp[i] will store the maximum length of non-decreasing subarray ending at index i
        dp = [1] * n
        
        # Initialize the maximum length as 1 (at least one element itself is non-decreasing)
        max_length = 1
        
        for i in range(1, n):
            if nums[i] >= nums[i - 1]:
                dp[i] = dp[i - 1] + 1
                max_length = max(max_length, dp[i])
        
        return max_length