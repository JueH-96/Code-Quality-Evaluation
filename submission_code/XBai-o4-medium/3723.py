from typing import List

class Solution:
    def sumOfGoodNumbers(self, nums: List[int], k: int) -> int:
        total = 0
        n = len(nums)
        for i in range(n):
            is_good = True
            left = i - k
            if 0 <= left < n:
                if nums[i] <= nums[left]:
                    is_good = False
            right = i + k
            if 0 <= right < n:
                if nums[i] <= nums[right]:
                    is_good = False
            if is_good:
                total += nums[i]
        return total