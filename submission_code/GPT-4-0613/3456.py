class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        dp = [[0]*(k+1) for _ in range(n+1)]
        for i in range(n-1, -1, -1):
            for j in range(k+1):
                dp[i][j] = max(dp[i+1][j], 1 + dp[i+1][j-1] if j > 0 and i+1 < n and nums[i] != nums[i+1] else 0)
        return max(dp[0])