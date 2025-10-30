from typing import List

class Solution:
    def longestSubsequence(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 2:
            return n
        
        # dp[i] will hold the length of the longest valid subsequence ending at index i
        dp = [1] * n
        
        # To store the maximum length of subsequence found
        max_length = 1
        
        # Iterate through each pair of indices to find valid subsequences
        for i in range(n):
            for j in range(i):
                # Calculate the absolute difference
                diff1 = abs(nums[i] - nums[j])
                
                # We need to check if we can form a valid subsequence
                if dp[j] > 1:  # Only consider j where we have a valid subsequence
                    # Check if we can extend the subsequence
                    last_diff = abs(nums[j] - nums[j-1]) if j > 0 else float('inf')
                    if diff1 <= last_diff:
                        dp[i] = max(dp[i], dp[j] + 1)
                
                # Update max_length
                max_length = max(max_length, dp[i])
        
        return max_length