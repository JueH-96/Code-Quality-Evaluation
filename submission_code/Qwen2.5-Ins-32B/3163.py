from typing import List

class Solution:
    def sumCounts(self, nums: List[int]) -> int:
        total_sum = 0
        n = len(nums)
        for i in range(n):
            distinct_values = set()
            for j in range(i, n):
                distinct_values.add(nums[j])
                total_sum += len(distinct_values) ** 2
        return total_sum