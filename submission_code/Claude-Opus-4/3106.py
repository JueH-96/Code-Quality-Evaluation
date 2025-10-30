class Solution:
    def lengthOfLongestSubsequence(self, nums: List[int], target: int) -> int:
        # dp[i] = maximum length of subsequence that sums to i
        # Initialize with -1 to indicate impossible sums
        dp = [-1] * (target + 1)
        dp[0] = 0  # Empty subsequence sums to 0
        
        # Process each number in nums
        for num in nums:
            # Traverse from right to left to avoid using the same element multiple times
            for sum_val in range(target, num - 1, -1):
                if dp[sum_val - num] != -1:
                    dp[sum_val] = max(dp[sum_val], dp[sum_val - num] + 1)
        
        return dp[target]