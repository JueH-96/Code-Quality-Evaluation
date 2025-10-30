import math
from typing import List

class Solution:
    def findValueOfPartition(self, nums: List[int]) -> int:
        # Step 1: Sort the array.
        # Sorting is essential because the minimum difference between max(nums1) and min(nums2)
        # will always occur between two adjacent elements in the sorted array.
        # This is because any larger gap (s_j - s_i where s_j and s_i are not adjacent in sorted_nums)
        # can be broken down into a sum of smaller adjacent gaps, and thus will be
        # greater than or equal to the smallest adjacent gap.
        nums.sort()
        
        # Step 2: Initialize min_diff to a very large value.
        # We use math.inf to ensure any valid difference will be smaller.
        min_diff = math.inf
        
        # Step 3: Iterate through the sorted array to find the minimum difference
        # between adjacent elements.
        # The loop runs from the first element up to the second-to-last element.
        # nums[i+1] will always be a valid index.
        for i in range(len(nums) - 1):
            # Calculate the difference between the current element and the next element.
            # Since the array is sorted, nums[i+1] >= nums[i], so the difference is non-negative.
            current_diff = nums[i+1] - nums[i]
            
            # Update min_diff if the current difference is smaller.
            min_diff = min(min_diff, current_diff)
            
            # Optimization: If min_diff becomes 0, it means there are duplicate numbers
            # in the array. In this case, we can partition the array such that
            # max(nums1) and min(nums2) are both equal to that duplicate number,
            # resulting in a partition value of 0. This is the absolute minimum possible,
            # so we can return immediately.
            if min_diff == 0:
                return 0
                
        # Step 4: After checking all adjacent pairs, min_diff will hold the
        # smallest possible value of the partition.
        return int(min_diff) # Convert math.inf (float) back to int in case it was not updated, though problem constraints ensure it will be updated
                             # No, min_diff will be an int because current_diff is int, and inf will convert to float automatically.
                             # It's fine to return float if math.inf, but constraints ensure N >= 2, so min_diff will be updated.
                             # So, it will always be an integer value in the end.