class Solution:
    def maximumTotalCost(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [0] * (n + 1)
        
        for i in range(n - 1, -1, -1):
            curr_sum = 0
            for j in range(i, n):
                curr_sum += nums[j] * (-1) ** (j - i)
                dp[i] = max(dp[i], curr_sum + dp[j + 1])
        
        return dp[0]