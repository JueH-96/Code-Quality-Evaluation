from typing import List

class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        memo = {}

        def solve(index, last_element, diff_count):
            if diff_count > k:
                return -float('inf')
            if index == len(nums):
                return 0
            if (index, last_element, diff_count) in memo:
                return memo[(index, last_element, diff_count)]

            res1 = solve(index + 1, last_element, diff_count)  # Don't include nums[index]
            res2 = -float('inf')
            # Include nums[index]
            if last_element is None:
                res2 = 1 + solve(index + 1, nums[index], diff_count)
            else:
                if nums[index] == last_element:
                    res2 = 1 + solve(index + 1, last_element, diff_count)
                else:
                    if diff_count + 1 <= k:
                        res2 = 1 + solve(index + 1, nums[index], diff_count + 1)

            result = max(res1, res2)
            memo[(index, last_element, diff_count)] = result
            return result

        max_len = solve(0, None, 0)
        return max(0, max_len)