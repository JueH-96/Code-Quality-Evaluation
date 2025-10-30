class Solution:
    def maximumJumps(self, nums: List[int], target: int) -> int:
        n = len(nums)
        # dp[i] will store the maximum number of jumps to reach index i
        # from index 0. If dp[i] is -1, index i is not reachable.
        dp = [-1] * n
        dp[0] = 0
        
        for i in range(n):
            # If current index is not reachable, skip
            if dp[i] == -1:
                continue
            # Try to jump to all possible j > i
            for j in range(i + 1, n):
                # Check if the difference condition is satisfied
                if abs(nums[j] - nums[i]) <= target:
                    dp[j] = max(dp[j], dp[i] + 1)
        
        return dp[n - 1]