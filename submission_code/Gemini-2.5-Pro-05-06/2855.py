from typing import List

class Solution:
    def maximumJumps(self, nums: List[int], target: int) -> int:
        n = len(nums)
        
        # dp[i] stores the maximum number of jumps to reach index i.
        # Initialize with -1 to signify "unreachable".
        # Minimum possible jumps is 0 (for dp[0]). Max possible jumps is n-1.
        dp = [-1] * n
        
        # Base case: We are at index 0 initially, having made 0 jumps.
        dp[0] = 0
        
        # Fill the dp table.
        # For each index i from 1 to n-1:
        for i in range(1, n):
            # Consider all previous indices j (0 to i-1) as potential previous steps.
            for j in range(i):
                # If index j is reachable from index 0 (i.e., dp[j] is not -1).
                if dp[j] != -1:
                    # Check if a jump from index j to index i is allowed.
                    # Condition: -target <= nums[i] - nums[j] <= target.
                    # Since target >= 0, this is equivalent to abs(nums[i] - nums[j]) <= target.
                    diff = nums[i] - nums[j] # Calculate difference once for clarity and minor efficiency
                    
                    if -target <= diff <= target:
                        # If we can jump from j to i:
                        # The number of jumps to reach i via j would be dp[j] + 1.
                        # We want to maximize dp[i].
                        # If dp[i] is currently -1 (unreachable), any valid path will update it.
                        # If dp[i] is already set by another path, update if this path is longer.
                        dp[i] = max(dp[i], dp[j] + 1)
                        
        # The value dp[n-1] holds the maximum number of jumps to reach the last index.
        # If dp[n-1] is still -1, it means the last index is unreachable.
        return dp[n-1]