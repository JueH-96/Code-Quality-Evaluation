from typing import List

class Solution:
    def maximumJumps(self, nums: List[int], target: int) -> int:
        n = len(nums)
        
        # dp[i] will store the maximum number of jumps to reach index i from index 0.
        # Initialize with -1 to indicate that index i is not reachable yet.
        dp = [-1] * n
        
        # Base case: It takes 0 jumps to be at the starting index 0.
        dp[0] = 0
        
        # Iterate through each index j from 1 to n-1
        # We want to calculate the maximum jumps to reach j.
        for j in range(1, n):
            # To reach index j, we must have come from some previous index i (0 <= i < j).
            # We check all possible previous indices i.
            for i in range(j):
                # Check if a jump is possible from index i to index j according to the rules.
                # The condition is -target <= nums[j] - nums[i] <= target.
                # Also, index i must be reachable from index 0 (i.e., dp[i] != -1).
                if -target <= nums[j] - nums[i] <= target and dp[i] != -1:
                    # If a jump from reachable i to j is possible, we can potentially reach j
                    # in dp[i] + 1 jumps.
                    # We want the maximum number of jumps to reach j, so we update dp[j]
                    # with the maximum of its current value and dp[i] + 1.
                    # The max function correctly handles the initial -1 in dp[j].
                    dp[j] = max(dp[j], dp[i] + 1)
        
        # The final answer is the maximum number of jumps to reach the last index (n-1).
        return dp[n - 1]