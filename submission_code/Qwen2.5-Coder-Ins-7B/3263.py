class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [[float('inf')] * 3 for _ in range(n)]
        dp[0][0] = nums[0]
        dp[1][1] = nums[0] + nums[1]
        dp[2][2] = nums[0] + nums[1] + nums[2]
        
        for i in range(3, n):
            dp[i][0] = nums[0]
            dp[i][1] = min(dp[i-1][0] + nums[i], dp[i][1])
            dp[i][2] = min(dp[i-1][1] + nums[i], dp[i][2])
        
        return dp[n-1][2]