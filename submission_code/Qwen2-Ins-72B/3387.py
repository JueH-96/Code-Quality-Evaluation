from typing import List

class Solution:
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        median_index = n // 2
        operations = 0

        if n % 2 == 1:
            # Odd number of elements, median is at the middle index
            operations += abs(nums[median_index] - k)
        else:
            # Even number of elements, median is the average of the two middle elements
            operations += abs(nums[median_index - 1] - k)
            operations += abs(nums[median_index] - k)

        return operations