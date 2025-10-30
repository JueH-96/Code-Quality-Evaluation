from typing import List

class Solution:
    def minimumAverage(self, nums: List[int]) -> float:
        averages = []
        current_nums = list(nums)
        for _ in range(len(nums) // 2):
            current_min = min(current_nums)
            current_max = max(current_nums)
            avg = (current_min + current_max) / 2
            averages.append(avg)
            current_nums.remove(current_min)
            current_nums.remove(current_max)
        return min(averages)