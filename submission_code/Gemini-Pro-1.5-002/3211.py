class Solution:
    def findMaximumLength(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [1] * n
        for i in range(1, n):
            for j in range(i):
                sub_sum = sum(nums[j:i+1])
                if sub_sum >= (nums[j-1] if j > 0 else 0):
                    dp[i] = max(dp[i], (dp[j-1] if j > 0 else 0) + 1)
        return max(dp)