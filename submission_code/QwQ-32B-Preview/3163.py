from typing import List

class Solution:
    def sumCounts(self, nums: List[int]) -> int:
        sum_total = 0
        n = len(nums)
        for i in range(n):
            current_set = set()
            for j in range(i, n):
                current_set.add(nums[j])
                distinct_count = len(current_set)
                sum_total += distinct_count ** 2
        return sum_total