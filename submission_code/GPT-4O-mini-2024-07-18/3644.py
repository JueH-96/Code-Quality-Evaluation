from typing import List

class Solution:
    def minimumSumSubarray(self, nums: List[int], l: int, r: int) -> int:
        n = len(nums)
        min_sum = float('inf')
        found_positive_sum = False
        
        # Check all subarrays of length from l to r
        for length in range(l, r + 1):
            for start in range(n - length + 1):
                subarray_sum = sum(nums[start:start + length])
                if subarray_sum > 0:
                    found_positive_sum = True
                    min_sum = min(min_sum, subarray_sum)
        
        return min_sum if found_positive_sum else -1