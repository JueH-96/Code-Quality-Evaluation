from typing import List

class Solution:
    def getFinalState(self, nums: List[int], k: int, multiplier: int) -> List[int]:
        nums_copy = list(nums)
        for _ in range(k):
            min_val = min(nums_copy)
            idx = nums_copy.index(min_val)
            nums_copy[idx] = min_val * multiplier
        return nums_copy