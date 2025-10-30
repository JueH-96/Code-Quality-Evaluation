from typing import List

class Solution:
    def sumCounts(self, nums: List[int]) -> int:
        sum_result = 0
        n = len(nums)
        for i in range(n):
            for j in range(i, n):
                current_sub = nums[i:j+1]
                distinct = len(set(current_sub))
                sum_result += distinct ** 2
        return sum_result