from typing import List

class Solution:
    def minimumAverage(self, nums: List[int]) -> float:
        # Sort the array so that we can pair smallest with largest
        nums.sort()
        n = len(nums)
        
        # Initialize the minimum average to a large value
        min_avg = float('inf')
        
        # Pair the i-th smallest with the i-th largest
        for i in range(n // 2):
            avg = (nums[i] + nums[n - 1 - i]) / 2.0
            if avg < min_avg:
                min_avg = avg
        
        return min_avg