class Solution:
    def maximumJumps(self, nums: List[int], target: int) -> int:
        n = len(nums)
        # dp[i] represents the maximum number of jumps to reach index i
        # -1 means unreachable
        dp = [-1] * n
        dp[0] = 0  # Starting position requires 0 jumps
        
        for i in range(n):
            if dp[i] == -1:  # If current position is unreachable, skip
                continue
                
            # Try to jump from position i to all valid positions j
            for j in range(i + 1, n):
                diff = nums[j] - nums[i]
                if -target <= diff <= target:  # Valid jump condition
                    dp[j] = max(dp[j], dp[i] + 1)
        
        return dp[n - 1]