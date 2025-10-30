from typing import List

class Solution:
    def minimumAverage(self, nums: List[int]) -> float:
        # Sort the list to easily pair smallest with largest.
        nums.sort()
        n = len(nums)
        averages = []
        
        # Pair the i-th smallest with the i-th largest.
        for i in range(n // 2):
            # Calculate the average of the i-th smallest and the i-th largest.
            avg = (nums[i] + nums[n - 1 - i]) / 2.0
            averages.append(avg)
        
        # Return the minimum average computed.
        return min(averages)