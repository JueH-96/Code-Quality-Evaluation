from typing import List

class Solution:
    def sumCounts(self, nums: List[int]) -> int:
        total = 0
        n = len(nums)
        for i in range(n):
            for j in range(i, n):
                current_set = set(nums[i:j+1])
                count = len(current_set)
                total += count * count
        return total