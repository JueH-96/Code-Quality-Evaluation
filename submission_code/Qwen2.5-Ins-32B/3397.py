from typing import List

class Solution:
    def addedInteger(self, nums1: List[int], nums2: List[int]) -> int:
        # Calculate the difference between the first elements of nums1 and nums2
        x = nums2[0] - nums1[0]
        
        # Verify that the same difference applies to all elements
        for i in range(len(nums1)):
            if nums1[i] + x != nums2[i]:
                return -1  # This case should not happen according to the problem constraints
        
        return x