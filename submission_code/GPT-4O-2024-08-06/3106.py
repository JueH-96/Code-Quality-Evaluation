from typing import List

class Solution:
    def lengthOfLongestSubsequence(self, nums: List[int], target: int) -> int:
        n = len(nums)
        dp = [{} for _ in range(n + 1)]
        
        # Initialize the dp array
        dp[0][0] = 0
        
        for i in range(1, n + 1):
            num = nums[i - 1]
            for sum_so_far in dp[i - 1]:
                # Case 1: Do not include the current number
                if sum_so_far in dp[i]:
                    dp[i][sum_so_far] = max(dp[i][sum_so_far], dp[i - 1][sum_so_far])
                else:
                    dp[i][sum_so_far] = dp[i - 1][sum_so_far]
                
                # Case 2: Include the current number
                new_sum = sum_so_far + num
                if new_sum <= target:
                    if new_sum in dp[i]:
                        dp[i][new_sum] = max(dp[i][new_sum], dp[i - 1][sum_so_far] + 1)
                    else:
                        dp[i][new_sum] = dp[i - 1][sum_so_far] + 1
        
        # Check for the longest subsequence that sums to target
        if target in dp[n]:
            return dp[n][target]
        else:
            return -1