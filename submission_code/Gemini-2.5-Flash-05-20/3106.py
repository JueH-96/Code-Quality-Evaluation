from typing import List

class Solution:
    def lengthOfLongestSubsequence(self, nums: List[int], target: int) -> int:
        # dp[s] will store the maximum length of a subsequence that sums up to 's'.
        # Initialize dp array with -1. This value signifies that a particular sum 's'
        # is currently unreachable by any subsequence.
        dp = [-1] * (target + 1)
        
        # Base case: An empty subsequence sums to 0 and has a length of 0.
        dp[0] = 0

        # Iterate through each number in the input array 'nums'.
        # Each 'num' represents an item that we can choose to include or not include
        # in our subsequence.
        for num in nums:
            # Iterate backwards from 'target' down to 'num'.
            # This backward iteration is critical for this 0-1 knapsack type problem.
            # It ensures that each 'num' from the input array is used at most once
            # in forming a subsequence. If we were to iterate forwards (from 'num' to 'target'),
            # when computing dp[s], dp[s - num] might have already been updated using
            # the *current* 'num', leading to 'num' being used multiple times for the
            # same subsequence construction.
            for s in range(target, num - 1, -1):
                # If 's - num' is a reachable sum (i.e., dp[s - num] is not -1),
                # it means we have found a subsequence that sums to 's - num'.
                if dp[s - num] != -1:
                    # By adding the current 'num' to this subsequence, we can form a sum 's'.
                    # The length of this new subsequence would be dp[s - num] + 1.
                    # We want the maximum possible length for sum 's', so we take the
                    # maximum between the current value of dp[s] and this newly calculated length.
                    dp[s] = max(dp[s], dp[s - num] + 1)

        # After processing all numbers, dp[target] will hold the maximum length
        # of a subsequence that sums up to 'target'.
        # If dp[target] is still -1, it means no such subsequence exists.
        return dp[target]