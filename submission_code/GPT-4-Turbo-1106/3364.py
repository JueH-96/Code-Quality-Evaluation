class Solution:
    def minimumValueSum(self, nums: list[int], andValues: list[int]) -> int:
        n, m = len(nums), len(andValues)
        dp = [[float('inf')] * (m + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        
        for i in range(1, n + 1):
            for j in range(1, min(i, m) + 1):
                and_so_far = nums[i - 1]
                for k in range(i, 0, -1):
                    and_so_far &= nums[k - 1]
                    if and_so_far == andValues[j - 1]:
                        dp[i][j] = min(dp[i][j], dp[k - 1][j - 1] + nums[i - 1])
        
        return dp[n][m] if dp[n][m] != float('inf') else -1