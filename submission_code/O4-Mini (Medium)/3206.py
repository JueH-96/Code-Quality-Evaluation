from typing import List

class Solution:
    def findIntersectionValues(self, nums1: List[int], nums2: List[int]) -> List[int]:
        # Create sets for quick membership checks
        set1 = set(nums1)
        set2 = set(nums2)
        
        # Count how many elements in nums1 appear in nums2
        count1 = sum(1 for x in nums1 if x in set2)
        # Count how many elements in nums2 appear in nums1
        count2 = sum(1 for x in nums2 if x in set1)
        
        return [count1, count2]