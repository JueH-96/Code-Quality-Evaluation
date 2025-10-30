class Solution:
    def maximumStrength(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]

        @lru_cache(None)
        def dp(i: int, k: int) -> int:
            if k == 0:
                return 0
            if i < 0:
                return -inf
            return max(dp(i - 1, k), dp(i - 2, k - 1) + nums[i] * (prefix_sum[i + 1] - prefix_sum[i]))

        return dp(n - 1, k)