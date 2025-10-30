from typing import List

class Solution:
    def sumOfGoodNumbers(self, nums: List[int], k: int) -> int:
        n = len(nums)
        total = 0
        
        for i in range(n):
            good = True
            if i - k >= 0:
                if not (nums[i] > nums[i - k]):
                    good = False
            if i + k < n:
                if not (nums[i] > nums[i + k]):
                    good = False
            if good:
                total += nums[i]
        return total