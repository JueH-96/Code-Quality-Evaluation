class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        n = len(nums)
        dp = [[-float('inf')] * 2 for _ in range(n)]
        dp[0][nums[0] % 2] = nums[0]
        for i in range(1, n):
            for j in range(i):
                if nums[i] % 2 == nums[j] % 2:
                    dp[i][nums[i] % 2] = max(dp[i][nums[i] % 2], dp[j][nums[j] % 2] + nums[i])
                else:
                    dp[i][nums[i] % 2] = max(dp[i][nums[i] % 2], dp[j][nums[j] % 2] + nums[i] - x)
            dp[i][nums[i] % 2] = max(dp[i][nums[i] % 2], nums[i])
        
        ans = nums[0]
        for i in range(1, n):
            ans = max(ans, dp[i][0], dp[i][1])

        return ans