from typing import List

class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        max_length = -1
        for i in range(len(nums)):
            for j in range(i + 2, len(nums) + 1):
                subarray = nums[i:j]
                if self.is_alternating(subarray):
                    max_length = max(max_length, len(subarray))
        return max_length

    def is_alternating(self, subarray: List[int]) -> bool:
        if len(subarray) < 2:
            return False
        diff = subarray[1] - subarray[0]
        if diff != 1:
            return False
        for i in range(2, len(subarray)):
            if subarray[i] - subarray[i - 1] != (-1) ** i * diff:
                return False
        return True