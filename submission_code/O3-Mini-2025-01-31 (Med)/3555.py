from typing import List

class Solution:
    def getFinalState(self, nums: List[int], k: int, multiplier: int) -> List[int]:
        for _ in range(k):
            # Find the minimum value in nums
            min_val = min(nums)
            # Find the index of the first occurrence of the minimum value
            idx = nums.index(min_val)
            # Multiply the element at this index by multiplier
            nums[idx] = min_val * multiplier
        return nums