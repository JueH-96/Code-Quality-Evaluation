from typing import List

class Solution:
    def minLengthAfterRemovals(self, nums: List[int]) -> int:
        n = len(nums)
        i, j = 0, (n + 1) // 2
        while i < n // 2 and j < n:
            if nums[i] < nums[j]:
                i += 1
            j += 1
        return n - 2 * i