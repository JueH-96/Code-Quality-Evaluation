class Solution:
    def maximumJumps(self, nums: List[int], target: int) -> int:
        n = len(nums)
        # dp[i] = maximum number of jumps to reach index i from index 0
        dp = [-1] * n
        dp[0] = 0  # Base case: no jumps needed to stay at index 0
        
        for i in range(n):
            if dp[i] == -1:  # Skip positions we can't reach
                continue
                
            for j in range(i + 1, n):
                # Check if jump from i to j is valid based on the difference constraint
                if -target <= nums[j] - nums[i] <= target:
                    # Update dp[j] with the maximum number of jumps
                    dp[j] = max(dp[j], dp[i] + 1)
        
        return dp[n-1]  # Return the maximum jumps to reach the last index