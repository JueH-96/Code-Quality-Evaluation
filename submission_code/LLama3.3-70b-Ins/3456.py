from typing import List

class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        memo = {}

        def dp(i, k, prev):
            if i == n:
                return 0
            if (i, k, prev) in memo:
                return memo[(i, k, prev)]
            take = 0
            if nums[i] == prev or k > 0:
                take = 1 + dp(i + 1, k - (prev != nums[i]), nums[i])
            skip = dp(i + 1, k, prev)
            memo[(i, k, prev)] = max(take, skip)
            return memo[(i, k, prev)]

        return dp(0, k, None)