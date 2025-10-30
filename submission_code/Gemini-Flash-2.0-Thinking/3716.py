class Solution:
    def longestSubsequence(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0

        dp = [{} for _ in range(n)]
        max_len = 0

        for i in range(n):
            dp[i][float('inf')] = 1
            max_len = max(max_len, 1)

            for j in range(i):
                diff = abs(nums[i] - nums[j])
                for prev_diff, length in dp[j].items():
                    if diff <= prev_diff:
                        dp[i][diff] = max(dp[i].get(diff, 0), length + 1)
                        max_len = max(max_len, length + 1)

        return max_len