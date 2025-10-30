class Solution:
    def canAliceWin(self, nums: List[int]) -> bool:
        total = sum(nums)
        dp = [0] * (total // 2 + 1)
        for num in nums:
            for i in range(total // 2, num - 1, -1):
                dp[i] = max(dp[i], dp[i - num] + num)
        return dp[total // 2] > total - dp[total // 2]