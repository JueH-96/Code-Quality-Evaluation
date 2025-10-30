from typing import List

class Solution:
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        # Compute sum of fixed (non-zero) elements and count the zeros in each array.
        s1 = sum(x for x in nums1 if x != 0)
        z1 = nums1.count(0)
        s2 = sum(x for x in nums2 if x != 0)
        z2 = nums2.count(0)
        
        # When there are no zeros in both arrays, the sum is fixed.
        if z1 == 0 and z2 == 0:
            return s1 if s1 == s2 else -1
        
        # When one of the arrays is fixed (has no zeros), we must match that fixed sum.
        # For the other array with zeros, the minimum sum we can reach is its fixed sum plus 
        # the count of zeros (since every zero must be replaced by a positive integer at least 1).
        if z1 == 0:
            # nums1 is fixed to s1. nums2 has flexibility.
            # The smallest possible sum for nums2 is s2 + z2.
            return s1 if s1 >= s2 + z2 else -1
        
        if z2 == 0:
            # nums2 is fixed to s2. nums1 has flexibility.
            return s2 if s2 >= s1 + z1 else -1
        
        # If both arrays have zeros, then each array's sum can be adjusted from a minimum value:
        # For nums1, minimum possible sum is s1 + z1, and for nums2 is s2 + z2.
        # We can choose any sum S >= that minimum for each array.
        # To achieve equality while minimizing S, choose S = max(s1 + z1, s2 + z2)
        return max(s1 + z1, s2 + z2)