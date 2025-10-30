from typing import List
from collections import defaultdict

class Solution:
    def lengthOfLongestSubsequence(self, nums: List[int], target: int) -> int:
        # Dictionary to store the longest subsequence length for each sum
        dp = defaultdict(int)
        dp[0] = 0

        # Iterate through each number in the array
        for num in nums:
            # Create a copy of the current dictionary to avoid modifying it while iterating
            current_dp = dict(dp)
            for current_sum in current_dp:
                new_sum = current_sum + num
                # Update the dictionary with the new sum and the length of the subsequence
                if new_sum not in dp or dp[new_sum] < dp[current_sum] + 1:
                    dp[new_sum] = dp[current_sum] + 1

        # Check if the target sum exists in the dictionary
        if target in dp:
            return dp[target]
        else:
            return -1