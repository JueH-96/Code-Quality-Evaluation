from typing import List

class Solution:
    def getFinalState(self, nums: List[int], k: int, multiplier: int) -> List[int]:
        """
        Repeatedly (k times) finds the earliest occurrence of the minimal element
        in `nums` and multiplies it by `multiplier`.
        """
        n = len(nums)
        for _ in range(k):
            # Find index of the first minimum value
            min_idx = 0
            min_val = nums[0]
            for i in range(1, n):
                if nums[i] < min_val:
                    min_val = nums[i]
                    min_idx = i
            # Apply the multiplier
            nums[min_idx] *= multiplier
        
        return nums