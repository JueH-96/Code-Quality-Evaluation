from typing import List

class Solution:
    def getFinalState(self, nums: List[int], k: int, multiplier: int) -> List[int]:
        for _ in range(k):
            min_val = min(nums)
            for idx, val in enumerate(nums):
                if val == min_val:
                    nums[idx] = val * multiplier
                    break
        return nums