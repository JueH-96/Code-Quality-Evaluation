class Solution:
    def maximumJumps(self, nums: List[int], target: int) -> int:
        n = len(nums)
        # Initialize a dp array where dp[i] represents the maximum number of jumps to reach index i
        dp = [-1] * n
        dp[0] = 0  # Starting point, no jumps needed to be at index 0

        # Iterate over each index i
        for i in range(n):
            if dp[i] == -1:
                continue  # If index i is not reachable, skip it

            # Try to jump from index i to any index j
            for j in range(i + 1, n):
                if -target <= nums[j] - nums[i] <= target:
                    dp[j] = max(dp[j], dp[i] + 1)

        return dp[-1]  # Return the maximum jumps to reach the last index