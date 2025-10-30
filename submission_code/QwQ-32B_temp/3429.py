from typing import List

class Solution:
    def isArraySpecial(self, nums: List[int]) -> bool:
        if len(nums) < 2:
            return True
        for i in range(len(nums) - 1):
            a = nums[i]
            b = nums[i+1]
            if (a % 2 + b % 2) != 1:
                return False
        return True