from typing import List

class Solution:
    def getFinalState(self, nums: List[int], k: int, multiplier: int) -> List[int]:
        result = [0] * len(nums)
        for i in range(k):
            min_val = min(nums)
            min_index = nums.index(min_val)
            result[i] = min_val * multiplier
            nums[min_index] = float('inf')
        return result