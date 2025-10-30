from typing import List

class Solution:
    def maximumJumps(self, nums: List[int], target: int) -> int:
        n = len(nums)
        # dp[i] stores the maximum number of jumps to reach index i.
        # Initialize with -1 (meaning unreachable).
        dp = [-1] * n
        dp[0] = 0  # starting at index 0 with 0 jumps
        
        # Iterate over every index and try all possible jumps from i to j.
        for i in range(n):
            if dp[i] == -1:
                continue  # this index is unreachable, so skip it
            for j in range(i + 1, n):
                # Check if the jump from index i to index j is valid:
                # Given the condition -target <= nums[j] - nums[i] <= target,
                # and since target is non-negative, this is equivalent to:
                # abs(nums[j] - nums[i]) <= target.
                if abs(nums[j] - nums[i]) <= target:
                    dp[j] = max(dp[j], dp[i] + 1)
        
        return dp[-1]