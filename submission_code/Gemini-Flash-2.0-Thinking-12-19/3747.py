from typing import List

class Solution:
    def maxAdjacentDistance(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Initialize max_diff. Since we are looking for the maximum absolute
        # difference, and absolute differences are non-negative, 0 is a safe
        # starting point. The constraints guarantee n >= 2, so there will
        # always be at least two elements to compare, resulting in a 
        # non-negative max_diff.
        max_diff = 0 

        # Iterate through adjacent elements in the standard linear order
        # from index 0 to n-2 (inclusive), comparing nums[i] and nums[i+1].
        for i in range(n - 1):
            current_diff = abs(nums[i] - nums[i+1])
            max_diff = max(max_diff, current_diff)

        # Check the difference between the last and first elements for circularity.
        # In a circular array, the element at index n-1 is adjacent to the
        # element at index 0.
        circular_diff = abs(nums[n-1] - nums[0])
        max_diff = max(max_diff, circular_diff)

        return max_diff