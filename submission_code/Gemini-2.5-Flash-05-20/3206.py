from typing import List

class Solution:
    def findIntersectionValues(self, nums1: List[int], nums2: List[int]) -> List[int]:
        # Convert nums2 to a set for efficient O(1) average time complexity lookups
        set2 = set(nums2)
        
        # Calculate the first value: number of indices i in nums1 such that nums1[i] occurs in nums2
        count1 = 0
        # Iterate through each element in nums1
        for num in nums1:
            # Check if the current element from nums1 is present in set2
            if num in set2:
                count1 += 1
                
        # Convert nums1 to a set for efficient O(1) average time complexity lookups
        set1 = set(nums1)
        
        # Calculate the second value: number of indices i in nums2 such that nums2[i] occurs in nums1
        count2 = 0
        # Iterate through each element in nums2
        for num in nums2:
            # Check if the current element from nums2 is present in set1
            if num in set1:
                count2 += 1
                
        # Return the two calculated counts as an array
        return [count1, count2]