from typing import List

class Solution:
    def minimumAverage(self, nums: List[int]) -> float:
        nums_sorted = sorted(nums)
        n = len(nums)
        averages = []
        for i in range(n // 2):
            avg = (nums_sorted[i] + nums_sorted[n - 1 - i]) / 2
            averages.append(avg)
        return min(averages)