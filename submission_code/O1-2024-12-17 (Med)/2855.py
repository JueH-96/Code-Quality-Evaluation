class Solution:
    def maximumJumps(self, nums: List[int], target: int) -> int:
        n = len(nums)
        # dp[i] will store the maximum number of jumps to reach index i, or -1 if unreachable
        dp = [-1] * n
        dp[0] = 0  # Start at index 0 with 0 jumps made
        
        for i in range(n):
            if dp[i] == -1:
                continue  # This index is not reachable
            for j in range(i + 1, n):
                # Check if we can jump from i to j
                if abs(nums[j] - nums[i]) <= target:
                    dp[j] = max(dp[j], dp[i] + 1)
        
        return dp[-1]