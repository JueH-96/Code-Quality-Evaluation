class Solution:
    def minimumValueSum(self, nums: List[int], andValues: List[int]) -> int:
        n, m = len(nums), len(andValues)
        dp = [[float('inf')] * (m + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        
        for i in range(1, n + 1):
            current_and = nums[i - 1]
            for j in range(1, min(i, m) + 1):
                for k in range(i - 1, j - 2, -1):
                    if current_and == andValues[j - 1]:
                        dp[i][j] = min(dp[i][j], dp[k][j - 1] + nums[i - 1])
                    current_and &= nums[k - 1]
        
        result = dp[n][m]
        return result if result != float('inf') else -1