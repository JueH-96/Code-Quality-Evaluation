from typing import List

class Solution:
    def lengthOfLongestSubsequence(self, nums: List[int], target: int) -> int:
        memo = {}
        n = len(nums)

        def solve(index, current_sum):
            if current_sum == target:
                return 0
            if current_sum > target:
                return -float('inf')
            if index == n:
                return -float('inf')

            if (index, current_sum) in memo:
                return memo[(index, current_sum)]

            exclude_len = solve(index + 1, current_sum)
            include_len = -float('inf')
            if current_sum + nums[index] <= target:
                include_len = 1 + solve(index + 1, current_sum + nums[index])

            result = max(exclude_len, include_len)
            memo[(index, current_sum)] = result
            return result

        result_len = solve(0, 0)
        if result_len < 0:
            return -1
        else:
            return result_len