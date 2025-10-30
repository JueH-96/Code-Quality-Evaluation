import math
from typing import List

class Solution:
    """
    This class provides a solution to find the length of the longest subsequence
    of an array `nums` that sums up to a given `target`.
    This problem is solved using dynamic programming, similar to the 0/1 Knapsack 
    problem or the subset sum problem, where the goal is to maximize the number 
    of elements (length of the subsequence) instead of maximizing value or just checking feasibility.
    """
    def lengthOfLongestSubsequence(self, nums: List[int], target: int) -> int:
        """
        Calculates the length of the longest subsequence of nums that sums to target
        using dynamic programming.

        Args:
          nums: A list of integers.
          target: The target sum.

        Returns:
          The length of the longest subsequence summing to target, or -1 if no such subsequence exists.
          
        Time Complexity: O(n * target), where n is the length of nums. The nested loops iterate 
                         through each number and then up to `target` potential sums.
        Space Complexity: O(target). We use a DP array of size `target + 1`.
        """
        
        # dp[j] will store the maximum length of a subsequence of the elements 
        # considered so far (from nums) that sums up to exactly j.
        # Initialize dp array of size target + 1 with -1.
        # -1 signifies that the sum j is currently unreachable. A value >= 0 indicates
        # a reachable sum, and the value itself is the maximum length found so far for that sum.
        dp = [-1] * (target + 1)
        
        # Base case: A sum of 0 can always be achieved with an empty subsequence,
        # which has a length of 0.
        dp[0] = 0
        
        # Iterate through each number in the input list nums.
        for num in nums:
            # Iterate backwards through the possible target sums j, from target down to num.
            # The backward iteration is crucial for the subsequence property (using each element at most once).
            # It ensures that when we calculate dp[j] using dp[j - num], the value dp[j - num] 
            # represents a subsequence formed *without* considering the current `num` in this specific update step 
            # related to the current outer loop iteration. This correctly models the 0/1 choice (either include 
            # the current `num` or not) for each element required for a subsequence.
            # If we iterated forwards (from num to target), dp[j] could be updated using a dp[j - num] 
            # value that might have already been updated using the *current* num in the same pass. This 
            # would incorrectly allow using the same element instance multiple times in the path leading to dp[j].
            for j in range(target, num - 1, -1):
                # Check if a valid subsequence summing to `j - num` exists.
                # `dp[j - num] != -1` (or equivalently `dp[j - num] >= 0` since lengths are non-negative) 
                # indicates that the sum `j - num` is reachable.
                if dp[j - num] != -1:
                    # If sum `j - num` is reachable with a subsequence of length `dp[j - num]`,
                    # then we can potentially form a subsequence summing to `j` by including the current number 'num'.
                    # The length of this new potential subsequence would be `dp[j - num] + 1`.
                    # We update `dp[j]` to be the maximum of its current value and this new length.
                    # This ensures `dp[j]` always stores the length of the *longest* subsequence found
                    # so far that sums to exactly `j`.
                    dp[j] = max(dp[j], dp[j - num] + 1)
                    
        # After iterating through all numbers in `nums`, `dp[target]` contains the maximum length
        # of a subsequence that sums exactly to `target`.
        # If `dp[target]` remains at its initial value of -1, it implies that no subsequence 
        # sums up to the target value.
        return dp[target]