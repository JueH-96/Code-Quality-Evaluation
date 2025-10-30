class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        nums.sort()
        dp = defaultdict(int)
        for x in nums:
            for y in [x, x + 1]:
                dp[y] = max(dp[y], dp[y - 1] + 1)
        return max(dp.values(), default=0)