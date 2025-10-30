class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [[0] * 2 for _ in range(n)]
        dp[0][0] = nums[0]
        dp[0][1] = nums[0]
        max_sum = nums[0]
        
        for i in range(1, n):
            dp[i][0] = max(dp[i-1][0] + nums[i], nums[i])
            dp[i][1] = max(dp[i-1][1] + nums[i], dp[i-1][0])
            max_sum = max(max_sum, dp[i][0], dp[i][1])
        
        return max_sum