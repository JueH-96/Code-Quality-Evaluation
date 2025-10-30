from typing import List

class Solution:
    def minimumAverage(self, nums: List[int]) -> float:
        nums_sorted = sorted(nums)
        averages = []
        n = len(nums_sorted)
        for _ in range(n // 2):
            min_val = nums_sorted[0]
            max_val = nums_sorted[-1]
            avg = (min_val + max_val) / 2
            averages.append(avg)
            nums_sorted = nums_sorted[1:-1]
        return min(averages)