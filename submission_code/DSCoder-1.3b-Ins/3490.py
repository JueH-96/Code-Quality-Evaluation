class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [1]*n
        for i in range(1, n):
            if nums[i] == nums[i-1]:
                dp[i] = 2
            else:
                dp[i] = dp[i-1] + 1
        max_len = max(dp)
        for i in range(n-1, -1, -1):
            if nums[i] == nums[i+1]:
                dp[i] = 2
            else:
                dp[i] = dp[i+1] + 1
                if dp[i] > max_len:
                    max_len = dp[i]
        return max_len