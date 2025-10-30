from typing import List

class Solution:
    def findIntersectionValues(self, nums1: List[int], nums2: List[int]) -> List[int]:
        # Convert both lists to sets for O(1) average membership checks
        set1, set2 = set(nums1), set(nums2)
        
        # Count positions in nums1 whose element appears in nums2
        count1 = sum(1 for x in nums1 if x in set2)
        
        # Count positions in nums2 whose element appears in nums1
        count2 = sum(1 for x in nums2 if x in set1)
        
        return [count1, count2]