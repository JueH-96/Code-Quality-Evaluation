class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        dp = [1] * n
        for i in range(1, n):
            if nums[i] == nums[i - 1] + 1:
                dp[i] = dp[i - 1] + 1
            elif nums[i] == nums[i - 1]:
                dp[i] = dp[i - 1]
        return max(dp)