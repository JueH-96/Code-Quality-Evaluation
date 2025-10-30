from typing import List

class Solution:
    def lengthOfLongestSubsequence(self, nums: List[int], target: int) -> int:
        n = len(nums)
        memo = {}

        def solve(index, remaining_target):
            if remaining_target == 0:
                return 0
            if index == n:
                return -float('inf')
            if remaining_target < 0:
                return -float('inf')
            if (index, remaining_target) in memo:
                return memo[(index, remaining_target)]

            option1 = -float('inf')
            if remaining_target >= nums[index]:
                option1 = 1 + solve(index + 1, remaining_target - nums[index])
            option2 = solve(index + 1, remaining_target)

            result = max(option1, option2)
            memo[(index, remaining_target)] = result
            return result

        longest_len = solve(0, target)
        if longest_len == -float('inf'):
            return -1
        else:
            return longest_len