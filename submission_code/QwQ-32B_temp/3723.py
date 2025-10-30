from typing import List

class Solution:
    def sumOfGoodNumbers(self, nums: List[int], k: int) -> int:
        total = 0
        n = len(nums)
        for i in range(n):
            valid = True
            # Check left index i - k
            left = i - k
            if left >= 0:
                if nums[i] <= nums[left]:
                    valid = False
            # Check right index i + k
            right = i + k
            if right < n:
                if nums[i] <= nums[right]:
                    valid = False
            if valid:
                total += nums[i]
        return total