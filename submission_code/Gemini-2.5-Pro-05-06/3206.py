from typing import List

class Solution:
    def findIntersectionValues(self, nums1: List[int], nums2: List[int]) -> List[int]:
        # Convert nums1 and nums2 to sets for efficient O(1) average time lookups.
        set_nums1 = set(nums1)
        set_nums2 = set(nums2)
        
        # Calculate the first value:
        # The number of indices i such that 0 <= i < n and nums1[i] occurs at least once in nums2.
        # This means iterating through each element of the original nums1 list
        # and checking if its value is present in set_nums2.
        # sum() treats True as 1 and False as 0.
        count1 = sum(num in set_nums2 for num in nums1)
        
        # Calculate the second value:
        # The number of indices i such that 0 <= i < m and nums2[i] occurs at least once in nums1.
        # Similarly, iterate through each element of the original nums2 list
        # and checking if its value is present in set_nums1.
        count2 = sum(num in set_nums1 for num in nums2)
        
        return [count1, count2]