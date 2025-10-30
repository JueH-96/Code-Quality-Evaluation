class Solution:
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        n = len(nums)
        dp = [[float('inf')] * 2 for _ in range(n)]
        dp[0][0] = nums[0]
        dp[0][1] = nums[0]
        for i in range(1, n):
            dp[i][0] = min(dp[i-1][0] & nums[i], dp[i-1][1] & nums[i])
            dp[i][1] = min(dp[i-1][0] | nums[i], dp[i-1][1] | nums[i])
        return min(dp[n-1][0], dp[n-1][1])