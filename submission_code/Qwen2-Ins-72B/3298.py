class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        nums.sort()
        dp = [[1 for _ in range(2)] for _ in range(len(nums))]
        max_len = 1
        for i in range(1, len(nums)):
            if nums[i] == nums[i-1]:
                dp[i][0] = dp[i-1][0] + 1
                dp[i][1] = dp[i-1][1]
            elif nums[i] == nums[i-1] + 1:
                dp[i][0] = dp[i-1][0]
                dp[i][1] = dp[i-1][1] + 1
            elif nums[i] == nums[i-1] + 2:
                dp[i][0] = dp[i-1][1]
                dp[i][1] = dp[i-1][0]
            else:
                dp[i][0] = 1
                dp[i][1] = 1
            max_len = max(max_len, max(dp[i][0], dp[i][1]))
        return max_len