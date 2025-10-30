from bisect import bisect_left

class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        dp = []
        max_sum = float('-inf')
        for i, num in enumerate(nums):
            if not dp or num - i >= dp[-1][0]:
                dp.append((num - i, num))
                max_sum = max(max_sum, num)
            else:
                idx = bisect_left(dp, (num - i, float('inf')))
                dp[idx] = (num - i, num)
                max_sum = max(max_sum, dp[idx][1])
        return max_sum