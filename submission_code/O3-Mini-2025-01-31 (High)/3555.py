from typing import List

class Solution:
    def getFinalState(self, nums: List[int], k: int, multiplier: int) -> List[int]:
        for _ in range(k):
            # Determine the minimum value in nums and get its first occurrence index.
            min_val = min(nums)
            idx = nums.index(min_val)
            # Update the found minimum value by multiplying with multiplier.
            nums[idx] = nums[idx] * multiplier
        return nums