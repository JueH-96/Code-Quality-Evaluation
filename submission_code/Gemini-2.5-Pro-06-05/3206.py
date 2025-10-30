from typing import List

class Solution:
    def findIntersectionValues(self, nums1: List[int], nums2: List[int]) -> List[int]:
        """
        Calculates two values based on the presence of elements between two arrays.

        1. The number of indices i in nums1 where nums1[i] is also in nums2.
        2. The number of indices j in nums2 where nums2[j] is also in nums1.

        Args:
            nums1: A list of integers.
            nums2: A list of integers.

        Returns:
            A list containing the two calculated values.
        """
        
        # Convert lists to sets for efficient O(1) average time membership checking.
        s1 = set(nums1)
        s2 = set(nums2)
        
        # Calculate the first value:
        # Sum 1 for each element in nums1 that is present in s2.
        # Iterating through the original list nums1 ensures that we count based on
        # indices, correctly handling any duplicate values as per the problem.
        count1 = sum(1 for num in nums1 if num in s2)
        
        # Calculate the second value:
        # Sum 1 for each element in nums2 that is present in s1.
        count2 = sum(1 for num in nums2 if num in s1)
        
        return [count1, count2]