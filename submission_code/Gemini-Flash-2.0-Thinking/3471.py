from typing import List

class Solution:
    def minimumAverage(self, nums: List[int]) -> float:
        averages = []
        n = len(nums)
        for _ in range(n // 2):
            min_element = min(nums)
            nums.remove(min_element)
            max_element = max(nums)
            nums.remove(max_element)
            average = (min_element + max_element) / 2.0
            averages.append(average)
        return min(averages)