from typing import List

class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        
        # Check if it's possible to satisfy both conditions
        if max(nums1) > max(nums2) and max(nums1) != nums1[-1]:
            return -1
        if max(nums2) > max(nums1) and max(nums2) != nums2[-1]:
            return -1
        
        # Initialize variables to store the maximum values and their indices
        max_val1 = max(nums1)
        max_val2 = max(nums2)
        max_idx1 = nums1.index(max_val1)
        max_idx2 = nums2.index(max_val2)
        
        # Initialize the count of operations
        operations = 0
        
        # If the maximum values are not at the end of their respective arrays
        if max_idx1 != n - 1 or max_idx2 != n - 1:
            # If the maximum values are at the same index, swap them
            if max_idx1 == max_idx2:
                nums1[-1], nums2[-1] = max_val1, max_val2
                operations += 1
            # If the maximum values are not at the same index, swap them with the values at the end of their respective arrays
            else:
                nums1[max_idx1], nums1[-1] = nums1[-1], nums1[max_idx1]
                nums2[max_idx2], nums2[-1] = nums2[-1], nums2[max_idx2]
                operations += 2
        
        # Check if the conditions are satisfied
        if nums1[-1] == max_val1 and nums2[-1] == max_val2:
            return operations
        else:
            return -1