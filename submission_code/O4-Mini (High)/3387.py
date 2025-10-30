from typing import List

class Solution:
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        # Sort the array so that the median is at index m = n//2
        nums.sort()
        n = len(nums)
        m = n // 2
        ops = 0

        # Any element at or before the median that is > k must be decreased down to k
        for i in range(m + 1):
            if nums[i] > k:
                ops += nums[i] - k

        # Any element at or after the median that is < k must be increased up to k
        for i in range(m, n):
            if nums[i] < k:
                ops += k - nums[i]

        return ops