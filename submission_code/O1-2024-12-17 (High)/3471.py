from typing import List

class Solution:
    def minimumAverage(self, nums: List[int]) -> float:
        averages = []
        while nums:
            smallest = min(nums)
            largest = max(nums)
            nums.remove(smallest)
            nums.remove(largest)
            avg = (smallest + largest) / 2
            averages.append(avg)
        return min(averages)