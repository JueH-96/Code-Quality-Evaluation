from typing import List

class Solution:
    def getFinalState(self, nums: List[int], k: int, multiplier: int) -> List[int]:
        for _ in range(k):
            current_min = min(nums)
            index = nums.index(current_min)
            nums[index] *= multiplier
        return nums