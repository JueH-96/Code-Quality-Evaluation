class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        from collections import defaultdict
        dp = defaultdict(int)
        for num in sorted(nums):
            dp[num] = max(dp[num], dp[num**0.5] + 1)
        return max(dp.values())