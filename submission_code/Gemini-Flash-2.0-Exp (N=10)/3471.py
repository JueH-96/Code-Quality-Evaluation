import heapq
from typing import List

class Solution:
    def minimumAverage(self, nums: List[int]) -> float:
        averages = []
        nums.sort()
        while nums:
            min_element = nums.pop(0)
            max_element = nums.pop()
            averages.append((min_element + max_element) / 2)
        return min(averages)