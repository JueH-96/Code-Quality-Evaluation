from typing import List

class Solution:
    def minimumAverage(self, nums: List[int]) -> float:
        averages = []
        n = len(nums)
        for _ in range(n // 2):
            min_num = min(nums)
            max_num = max(nums)
            average = (min_num + max_num) / 2
            averages.append(average)
            nums.remove(min_num)
            nums.remove(max_num)
        return min(averages)