class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [[1] * 2 for _ in range(n)]
        res = 2
        for i in range(1, n):
            for j in range(i):
                if (nums[i] + nums[j]) % 2 == 0:
                    dp[i][0] = max(dp[i][0], dp[j][0] + 1)
                    dp[i][1] = max(dp[i][1], dp[j][1] + 1)
                else:
                    dp[i][0] = max(dp[i][0], dp[j][1] + 1)
                    dp[i][1] = max(dp[i][1], dp[j][0] + 1)
            res = max(res, dp[i][0], dp[i][1])
        return res