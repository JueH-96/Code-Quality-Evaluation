class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        nums.sort()
        dp = [1]*len(nums)
        for i in range(1, len(nums)):
            if nums[i] == nums[i-1]:
                dp[i] = dp[i-1]
            elif nums[i] == nums[i-1] + 1:
                dp[i] = max(dp[i-1] + 1, dp[i])
            else:
                dp[i] = max(dp[i], dp[i-1])
        return max(dp)