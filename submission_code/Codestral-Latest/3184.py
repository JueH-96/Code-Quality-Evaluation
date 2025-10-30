class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        from sortedcontainers import SortedDict

        n = len(nums)
        dp = [0] * n
        sd = SortedDict()

        for i in range(n):
            dp[i] = nums[i]
            if nums[i] > 0:
                idx = sd.bisect_left(nums[i] - i)
                if idx > 0:
                    dp[i] += sd.peekitem(idx - 1)[1]
            sd[nums[i] - i] = dp[i]

        return max(dp)