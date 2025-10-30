import math
from typing import List

class Solution:
    """
    Solves the maximum jumps problem using dynamic programming.
    This problem can be viewed as finding the longest path in a Directed Acyclic Graph (DAG),
    where the nodes are the indices of the array and edges represent valid jumps.
    The length of each edge is 1 (representing one jump).
    """
    def maximumJumps(self, nums: List[int], target: int) -> int:
        """
        Calculates the maximum number of jumps to reach the end of the array (index n-1)
        starting from index 0.

        A jump from index i to index j (where 0 <= i < j < n) is valid if
        -target <= nums[j] - nums[i] <= target, which is equivalent to
        abs(nums[j] - nums[i]) <= target.

        Args:
            nums: A 0-indexed list of integers representing the values at each position.
            target: An integer representing the maximum allowed absolute difference 
                    between values for a valid jump.

        Returns:
            The maximum number of jumps required to reach index n - 1 from index 0.
            Returns -1 if index n - 1 is unreachable from index 0 following the jump rules.
        """
        n = len(nums)

        # dp[k] will store the maximum number of jumps needed to reach index k from index 0.
        # Initialize all elements to -1, signifying that these indices are initially unreachable.
        # A value of -1 indicates that the index cannot be reached from index 0.
        dp = [-1] * n

        # Base case: We start at index 0. 
        # It takes 0 jumps to reach the starting index itself.
        dp[0] = 0

        # Iterate through all possible destination indices j, starting from index 1 up to n-1.
        # We calculate the maximum jumps to reach dp[j] based on the reachable preceding indices.
        for j in range(1, n):
            # For each destination index j, iterate through all possible preceding indices i (0 <= i < j)
            # from which we could potentially make a jump to j.
            for i in range(j):
                # Check if index i is reachable from the start (index 0).
                # If dp[i] is -1, it means index i cannot be reached from index 0,
                # so we cannot use it as a starting point for a jump to j.
                if dp[i] != -1:
                    
                    # Check if the jump condition is satisfied: abs(nums[j] - nums[i]) <= target.
                    # This ensures the difference between the values at the jump endpoints is within the allowed range.
                    if abs(nums[j] - nums[i]) <= target:
                        # If a valid jump is possible from i to j, it means we can potentially reach index j
                        # from index i. The number of jumps to reach j via this path is
                        # the number of jumps to reach i (dp[i]) plus one more jump (from i to j).
                        
                        # We want to find the path with the maximum number of jumps to reach j.
                        # So, we update dp[j] with the maximum value found so far among all possible paths reaching j.
                        # If dp[j] is currently -1 (meaning j was previously considered unreachable), 
                        # it will be updated to dp[i] + 1 (the number of jumps for this first found path to j).
                        # If dp[j] already has a positive value (meaning j is reachable via some other path),
                        # we update it only if the current path (via i) offers more jumps (dp[i] + 1 > dp[j]).
                        dp[j] = max(dp[j], dp[i] + 1)

        # After iterating through all possible jumps and filling the dp table,
        # dp[n - 1] will contain the maximum number of jumps required to reach the last index (n - 1)
        # from the starting index 0.
        # If dp[n - 1] remains -1, it means the last index was never reached according to the rules.
        return dp[n - 1]