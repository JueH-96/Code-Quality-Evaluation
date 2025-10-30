class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        dp = [float('inf')] * n
        min_nums = nums[:]
        res = float('inf')

        for _ in range(n):
            for i in range(n):
                min_nums[i] = min(min_nums[i], nums[(i - _) % n])
                dp[i] = min(dp[i], min_nums[i] + _ * x)
            res = min(res, dp[_])

        return res