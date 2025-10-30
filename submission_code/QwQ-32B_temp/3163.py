from typing import List

class Solution:
    def sumCounts(self, nums: List[int]) -> int:
        total = 0
        n = len(nums)
        for i in range(n):
            for j in range(i, n):
                sub = nums[i:j+1]
                distinct = len(set(sub))
                total += distinct * distinct
        return total