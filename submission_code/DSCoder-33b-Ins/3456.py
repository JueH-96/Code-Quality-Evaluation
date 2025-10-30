class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        dp = [0] * (n + 1)
        max_len = 0
        for i in range(n - 1, -1, -1):
            if nums[i] <= k:
                dp[i] = dp[i + 1] + 1
            else:
                dp[i] = 0
            max_len = max(max_len, dp[i])
        return max_len