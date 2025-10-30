class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        n = len(nums)
        mod = 10**9 + 7
        dp = [[0] * 51 for _ in range(n)]
        for i in range(nums[0] + 1):
            dp[0][i] = 1
        for i in range(1, n):
            prefix_sum = [0] * 52
            for j in range(51):
                prefix_sum[j + 1] = (prefix_sum[j] + dp[i - 1][j]) % mod
            for j in range(nums[i] + 1):
                upper_bound = min(j, nums[i - 1] - (nums[i] - j))
                if upper_bound >= 0:
                    dp[i][j] = prefix_sum[upper_bound + 1]
        ans = 0
        for i in range(nums[n - 1] + 1):
            ans = (ans + dp[n - 1][i]) % mod
        return ans