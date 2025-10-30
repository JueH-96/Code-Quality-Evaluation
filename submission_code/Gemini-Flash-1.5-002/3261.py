class Solution:
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        n = len(nums)
        if k >= n - 1:
            return reduce(lambda x, y: x & y, nums)

        dp = {}

        def solve(idx, ops_left, current_nums):
            if ops_left == 0:
                return reduce(lambda x, y: x | y, current_nums)

            if (idx, ops_left, tuple(current_nums)) in dp:
                return dp[(idx, ops_left, tuple(current_nums))]

            min_or = float('inf')
            for i in range(len(current_nums) - 1):
                new_nums = current_nums[:i] + [current_nums[i] & current_nums[i+1]] + current_nums[i+2:]
                min_or = min(min_or, solve(0, ops_left - 1, new_nums))

            min_or = min(min_or, reduce(lambda x, y: x | y, current_nums))

            dp[(idx, ops_left, tuple(current_nums))] = min_or
            return min_or

        return solve(0, k, nums)

from functools import reduce