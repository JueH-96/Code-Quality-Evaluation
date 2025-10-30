from typing import List

class Solution:
    def findIntersectionValues(self, nums1: List[int], nums2: List[int]) -> List[int]:
        set_nums2 = set(nums2)
        set_nums1 = set(nums1)
        
        count1 = sum(1 for num in nums1 if num in set_nums2)
        count2 = sum(1 for num in nums2 if num in set_nums1)
        
        return [count1, count2]