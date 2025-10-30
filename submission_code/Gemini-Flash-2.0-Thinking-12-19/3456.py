from typing import List

class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        memo = {}

        def solve(index, last_val, diff_left):
            if index == len(nums):
                return 0
            if (index, last_val, diff_left) in memo:
                return memo[(index, last_val, diff_left)]

            # Option 1: Skip nums[index]
            res1 = solve(index + 1, last_val, diff_left)

            # Option 2: Include nums[index]
            res2 = 0
            if last_val is None:
                res2 = 1 + solve(index + 1, nums[index], diff_left)
            else:
                if nums[index] == last_val:
                    res2 = 1 + solve(index + 1, last_val, diff_left)
                elif diff_left > 0:
                    res2 = 1 + solve(index + 1, nums[index], diff_left - 1)
                else:
                    res2 = 0 # Not a good subsequence if we include and diff_left is 0 and nums[index] != last_val

            res = max(res1, res2)
            memo[(index, last_val, diff_left)] = res
            return res

        return solve(0, None, k)