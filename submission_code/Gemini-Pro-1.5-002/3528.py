class Solution:
    def findMaximumScore(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [0] * n
        for i in range(n - 2, -1, -1):
            for j in range(i + 1, n):
                score = (j - i) * nums[i]
                if j == n - 1:
                    dp[i] = max(dp[i], score)
                else:
                    dp[i] = max(dp[i], score + dp[j])
        return dp[0]