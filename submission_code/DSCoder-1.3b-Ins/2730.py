class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        n = len(nums)
        dp = [0] * (n + 1)
        prefix = [0] * (n + 1)
        for i in range(1, n + 1):
            dp[i] = dp[i - 1] | nums[i - 1]
            prefix[i] = prefix[i - 1] | nums[0:i]
        res = dp[n]
        for i in range(n):
            if i + k < n:
                res = max(res, dp[i] | dp[i + k + 1])
            if i - k >= 0:
                res = max(res, prefix[i - k] | nums[i - k + 1:n])
        return res