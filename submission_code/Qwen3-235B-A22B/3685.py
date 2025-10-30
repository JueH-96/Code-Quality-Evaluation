from typing import List

class Solution:
    def countSubarrays(self, nums: List[int]) -> int:
        count = 0
        n = len(nums)
        for i in range(n - 2):
            first = nums[i]
            second = nums[i + 1]
            third = nums[i + 2]
            if second % 2 != 0:
                continue
            half = second // 2
            if first + third == half:
                count += 1
        return count