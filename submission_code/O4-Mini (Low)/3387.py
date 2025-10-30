from typing import List

class Solution:
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        # Sort the array to identify the median position
        nums.sort()
        n = len(nums)
        m = n // 2     # median index, using the larger one for even lengths
        ops = 0
        
        # Elements to the left of the median must be <= k
        # If any is greater than k, decrease it to k
        for i in range(m):
            if nums[i] > k:
                ops += nums[i] - k
        
        # The median itself must become k
        ops += abs(nums[m] - k)
        
        # Elements to the right of the median must be >= k
        # If any is less than k, increase it to k
        for i in range(m+1, n):
            if nums[i] < k:
                ops += k - nums[i]
        
        return ops