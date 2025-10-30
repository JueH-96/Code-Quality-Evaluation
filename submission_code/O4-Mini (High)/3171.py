from typing import List

class Solution:
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        # Sum of current (non-zero) elements and count zeros in each array
        sum1 = sum(nums1)
        zeros1 = nums1.count(0)
        sum2 = sum(nums2)
        zeros2 = nums2.count(0)
        
        # The final equal sum S must at least be:
        #   sum1 + zeros1  (each zero replaced by at least 1)
        #   sum2 + zeros2
        S = max(sum1 + zeros1, sum2 + zeros2)
        
        # If an array has no zeros, its sum cannot change.
        # So if zeros1 == 0, we must have S == sum1; otherwise impossible.
        if zeros1 == 0 and S != sum1:
            return -1
        # Similarly for the second array.
        if zeros2 == 0 and S != sum2:
            return -1
        
        return S