from typing import List

class Solution:
    def addedInteger(self, nums1: List[int], nums2: List[int]) -> int:
        # Calculate the difference between the first elements of the two arrays
        x = nums2[0] - nums1[0]
        
        # Check if adding x to each element of nums1 results in nums2
        for i in range(len(nums1)):
            if nums1[i] + x != nums2[i]:
                return "No solution"
        
        # If the loop completes without finding a mismatch, return x
        return x