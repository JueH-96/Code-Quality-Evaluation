from typing import List

class Solution:
    def addedInteger(self, nums1: List[int], nums2: List[int]) -> int:
        # According to the problem statement, if an integer 'x' is added to each element of nums1,
        # the resulting array becomes equal to nums2 (meaning they contain the same integers
        # with the same frequencies).

        # This implies that the minimum element of the transformed nums1 must be equal to
        # the minimum element of nums2.

        # Let min_nums1 be the minimum value in the original nums1.
        # Let min_nums2 be the minimum value in nums2.

        # After adding 'x' to every element of nums1, the new minimum value in nums1 will be
        # min_nums1 + x.

        # Since the transformed nums1 is equivalent to nums2, we must have:
        # min_nums1 + x = min_nums2

        # We can solve for x:
        # x = min_nums2 - min_nums1

        # Find the minimum element in nums1
        min_nums1 = min(nums1)
        
        # Find the minimum element in nums2
        min_nums2 = min(nums2)
        
        # Calculate the integer x
        x = min_nums2 - min_nums1
        
        return x