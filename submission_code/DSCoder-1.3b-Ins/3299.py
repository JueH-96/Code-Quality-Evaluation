class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [1] * n
        for i in range(1, n):
            if nums[i] != nums[i-1]:
                dp[i] = dp[i-1] + 1
            else:
                j = i - 1
                while j >= 0 and nums[j] == nums[i]:
                    j -= 1
                if j != -1:
                    dp[i] = dp[j] * 2 - 1
                else:
                    dp[i] = 2
        return max(dp)