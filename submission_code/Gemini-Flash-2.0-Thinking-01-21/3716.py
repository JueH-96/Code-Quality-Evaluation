class Solution:
    def longestSubsequence(self, nums: List[int]) -> int:
        n = len(nums)
        if n <= 1:
            return n
        dp = [[0] * 301 for _ in range(n)]
        max_len = 1
        for i in range(n):
            dp[i][0] = 1

        for i in range(1, n):
            for j in range(i):
                diff = abs(nums[i] - nums[j])
                dp[i][diff] = max(dp[i][diff], 2)
                for prev_diff in range(301):
                    if dp[j][prev_diff] >= 2:
                        if diff <= prev_diff:
                            dp[i][diff] = max(dp[i][diff], dp[j][prev_diff] + 1)

        for i in range(n):
            for d in range(301):
                max_len = max(max_len, dp[i][d])

        return max_len