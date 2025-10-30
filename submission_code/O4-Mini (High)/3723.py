from typing import List

class Solution:
    def sumOfGoodNumbers(self, nums: List[int], k: int) -> int:
        total = 0
        n = len(nums)
        for i in range(n):
            good = True
            # Check left neighbor if it exists
            if i - k >= 0 and nums[i] <= nums[i - k]:
                good = False
            # Check right neighbor if it exists
            if i + k < n and nums[i] <= nums[i + k]:
                good = False
            if good:
                total += nums[i]
        return total