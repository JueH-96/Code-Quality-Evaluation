class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        n = len(nums)
        mod = 10**9 + 7
        dp = [[0] * 51 for _ in range(n + 1)]

        # Base case: i = 1
        for v in range(nums[0] + 1):
            dp[1][v] = 1

        for i in range(2, n + 1):
            for v in range(nums[i - 1] + 1):
                lower_bound_prev_v = 0
                upper_bound_prev_v = min(nums[i - 2], v - max(0, nums[i - 1] - nums[i - 2]))

                if upper_bound_prev_v >= 0:
                    for prev_v in range(upper_bound_prev_v + 1):
                        dp[i][v] = (dp[i][v] + dp[i - 1][prev_v]) % mod

        ans = 0
        for v in range(nums[n - 1] + 1):
            ans = (ans + dp[n][v]) % mod

        return ans