from typing import List

class Solution:
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        # Sort the array to identify the median position
        nums.sort()
        n = len(nums)
        m = n // 2  # median index (0-based); for even n this is the upper median
        
        ops = 0
        # Any element left of or at m that is greater than k must be decreased to k
        for i in range(m + 1):
            if nums[i] > k:
                ops += nums[i] - k
        
        # Any element right of or at m that is less than k must be increased to k
        for i in range(m, n):
            if nums[i] < k:
                ops += k - nums[i]
        
        return ops