from typing import List

class Solution:
    def minimumAverage(self, nums: List[int]) -> float:
        nums.sort()
        averages = []
        
        while nums:
            minElement = nums.pop(0)
            maxElement = nums.pop(-1)
            averages.append((minElement + maxElement) / 2)
        
        return min(averages)