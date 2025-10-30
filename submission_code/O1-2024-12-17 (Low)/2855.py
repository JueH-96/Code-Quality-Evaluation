class Solution:
    def maximumJumps(self, nums: List[int], target: int) -> int:
        n = len(nums)
        # We'll use a DP array where dp[i] represents the maximum number of jumps
        # needed to reach index i from index 0. If i is not reachable, dp[i] = -1.
        dp = [-1] * n
        dp[0] = 0  # We start at index 0, so the jump count to 0 is 0
        
        for i in range(n):
            if dp[i] == -1:
                continue  # If index i is not reachable, skip
            for j in range(i+1, n):
                if -target <= nums[j] - nums[i] <= target:
                    dp[j] = max(dp[j], dp[i] + 1)
        
        return dp[-1]