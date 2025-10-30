from typing import List

class Solution:
    def findIntersectionValues(self, nums1: List[int], nums2: List[int]) -> List[int]:
        # Convert the lists to sets for efficient lookups
        set1 = set(nums1)
        set2 = set(nums2)
        
        # Calculate the number of indices in nums1 where the element occurs in nums2
        count1 = sum(1 for num in nums1 if num in set2)
        
        # Calculate the number of indices in nums2 where the element occurs in nums1
        count2 = sum(1 for num in nums2 if num in set1)
        
        # Return the counts as a list
        return [count1, count2]