from typing import List

class Solution:
    def getFinalState(self, nums: List[int], k: int, multiplier: int) -> List[int]:
        for _ in range(k):
            # find the current minimum value and its first occurrence
            min_val = min(nums)
            idx = nums.index(min_val)
            # replace it by multiplying with the multiplier
            nums[idx] = nums[idx] * multiplier
        return nums