from typing import List

class Solution:
    def minimumAverage(self, nums: List[int]) -> float:
        # Sort nums so that we can pair the smallest with the largest, 
        # second smallest with second largest, and so on.
        sorted_nums = sorted(nums)
        n = len(sorted_nums)
        
        # Collect the average values of paired elements.
        averages = [(sorted_nums[i] + sorted_nums[n - 1 - i]) / 2 for i in range(n // 2)]
        
        # Return the minimum average from the list.
        return min(averages)