from typing import List

class Solution:
    def maximumJumps(self, nums: List[int], target: int) -> int:
        n = len(nums)
        
        # dp[i] will store the maximum number of jumps to reach index i from index 0.
        # Initialize all entries to -1, signifying that the index is currently unreachable.
        dp = [-1] * n
        
        # Base case: It takes 0 jumps to be at the starting index 0.
        dp[0] = 0
        
        # Iterate through each possible destination index 'j' from 1 to n-1.
        # We start from 1 because index 0 is our initial position and its dp value is set.
        for j in range(1, n):
            # For each 'j', iterate through all possible preceding indices 'i' (from 0 to j-1).
            # If we can reach 'i' and a valid jump exists from 'i' to 'j',
            # we consider this path to update dp[j].
            for i in range(j):
                # Condition 1: Check if index 'i' is reachable from index 0.
                # If dp[i] is -1, it means 'i' is unreachable, so we cannot jump from it.
                if dp[i] != -1:
                    # Condition 2: Check if a jump from 'i' to 'j' is valid according to the problem rules.
                    # The condition is -target <= nums[j] - nums[i] <= target,
                    # which is mathematically equivalent to abs(nums[j] - nums[i]) <= target.
                    if abs(nums[j] - nums[i]) <= target:
                        # If both conditions are met, a valid jump from 'i' to 'j' is possible.
                        # We want to find the maximum number of jumps to reach 'j'.
                        # So, we update dp[j] by taking the maximum between its current value
                        # and (dp[i] + 1), where dp[i] is the jumps to 'i' and 1 is the jump from 'i' to 'j'.
                        dp[j] = max(dp[j], dp[i] + 1)
                        
        # After filling the entire dp array, dp[n-1] will contain the maximum number of jumps
        # to reach the last index (n-1) from index 0.
        # If dp[n-1] is still -1, it signifies that index n-1 is unreachable.
        # In this case, as per the problem statement, we should return -1.
        return dp[n-1]