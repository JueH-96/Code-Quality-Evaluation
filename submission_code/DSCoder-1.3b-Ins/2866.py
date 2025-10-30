class Solution:
    def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
        n = len(nums)
        dp = [0] * (n + 1)
        max_len = 0

        for i in range(n):
            if nums[i] % 2 == 0:
                dp[i + 1] = dp[i] + 1
                if nums[i] + dp[i] <= threshold:
                    max_len = max(max_len, dp[i + 1])
            else:
                dp[i + 1] = 1
                if nums[i] + dp[i] <= threshold:
                    max_len = max(max_len, dp[i + 1])

        return max_len