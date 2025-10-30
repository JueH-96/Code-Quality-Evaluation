class Solution:
    def maximumJumps(self, nums: List[int], target: int) -> int:
        n = len(nums)
        # dp[i] represents the maximum number of jumps to reach index i
        # Initialize with -1 to indicate unreachable positions
        dp = [-1] * n
        dp[0] = 0  # Starting position requires 0 jumps
        
        for i in range(1, n):
            for j in range(i):
                # Check if we can reach j and if we can jump from j to i
                if dp[j] != -1 and -target <= nums[i] - nums[j] <= target:
                    dp[i] = max(dp[i], dp[j] + 1)
        
        return dp[n - 1]