class Solution:
    def maximumTotalCost(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0

        dp = {}

        def solve(index):
            if index == -1:
                return 0
            if index in dp:
                return dp[index]

            # Option 1: The last subarray is [nums[index]]
            cost1 = solve(index - 1) + nums[index]

            # Option 2: The last subarray is [nums[index - 1], nums[index]]
            cost2 = -float('inf')
            if index > 0:
                cost2 = solve(index - 2) + nums[index - 1] - nums[index]

            dp[index] = max(cost1, cost2)
            return dp[index]

        return solve(n - 1)