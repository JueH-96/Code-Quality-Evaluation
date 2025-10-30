class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        nums.sort()
        dp = {}
        ans = 0
        for x in nums:
            init_x = dp.get(x, 0)
            dp[x] = max(init_x, dp.get(x-1, 0) + 1)
            dp[x+1] = max(dp.get(x+1, 0), init_x + 1)
            ans = max(ans, dp[x], dp[x+1])
        return ans