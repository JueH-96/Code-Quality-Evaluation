class Solution:
    def minCost(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [0] * (n + 5)
        if n >= 1:
            dp[n - 1] = nums[n - 1]
        if n >= 2:
            dp[n - 2] = max(nums[n - 2], nums[n - 1])
        for i in range(n - 3, -1, -1):
            op1 = max(nums[i], nums[i + 1]) + dp[i + 2]
            op2 = max(nums[i], nums[i + 2]) + dp[i + 3]
            op3 = max(nums[i + 1], nums[i + 2]) + dp[i + 3]
            dp[i] = min(op1, op2, op3)
        return dp[0]