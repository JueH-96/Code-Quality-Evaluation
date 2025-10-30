from typing import List

class Solution:
    def minimumAverage(self, nums: List[int]) -> float:
        # Sort the list so we can pair the smallest and largest elements directly
        nums.sort()
        n = len(nums)
        # Compute averages for each pair (i, n-1-i)
        averages = [(nums[i] + nums[n-1-i]) / 2.0 for i in range(n // 2)]
        # Return the minimum of these averages
        return min(averages)