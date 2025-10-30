from typing import List

class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        def backtrack(index, k, nums):
            if k == 0 or index == len(nums):
                return 0
            max_or = 0
            for i in range(index, len(nums)):
                # Try multiplying the current number by 2
                nums[i] *= 2
                # Recursively try all possibilities
                max_or = max(max_or, backtrack(index + 1, k - 1, nums) | nums[i])
                # Backtrack
                nums[i] //= 2
            return max_or

        max_or = 0
        for i in range(len(nums)):
            # Try multiplying the current number by 2
            nums[i] *= 2
            # Recursively try all possibilities
            max_or = max(max_or, backtrack(0, k - 1, nums) | nums[i])
            # Backtrack
            nums[i] //= 2

        return max_or