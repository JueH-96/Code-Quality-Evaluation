from typing import List

class Solution:
    def getFinalState(self, nums: List[int], k: int, multiplier: int) -> List[int]:
        # Perform k operations
        for _ in range(k):
            # Find the minimum value in nums and its first occurrence index
            min_val = min(nums)
            idx = nums.index(min_val)
            # Multiply the selected minimum by the multiplier
            nums[idx] = nums[idx] * multiplier
        return nums