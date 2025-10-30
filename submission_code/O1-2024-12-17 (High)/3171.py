from typing import List

class Solution:
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        # Calculate the sum of non-zero elements and the number of zeroes in each array
        sum1 = 0
        z1 = 0
        for x in nums1:
            if x == 0:
                z1 += 1
            else:
                sum1 += x
        
        sum2 = 0
        z2 = 0
        for x in nums2:
            if x == 0:
                z2 += 1
            else:
                sum2 += x
        
        # If both arrays have no zeroes, they are fixed. Check for equality.
        if z1 == 0 and z2 == 0:
            return sum1 if sum1 == sum2 else -1
        
        # If nums1 has no zeroes, its sum is fixed. We can only raise nums2 if z2 > 0.
        if z1 == 0:
            # We want sum2 + Y = sum1. Hence Y = sum1 - sum2.
            # We must have sum1 >= sum2 and sum1 - sum2 >= z2 (since we need at least z2 total from strictly positive replacements).
            if sum1 < sum2:
                return -1
            if (sum1 - sum2) < z2:
                return -1
            return sum1
        
        # If nums2 has no zeroes, its sum is fixed. We can only raise nums1 if z1 > 0.
        if z2 == 0:
            # We want sum1 + X = sum2. Hence X = sum2 - sum1.
            # We must have sum2 >= sum1 and sum2 - sum1 >= z1.
            if sum2 < sum1:
                return -1
            if (sum2 - sum1) < z1:
                return -1
            return sum2
        
        # If both arrays have zeroes, we can add to both sums.
        # Let d = sum2 - sum1, X = sum of replacements in nums1, Y = sum of replacements in nums2.
        # We need sum1 + X = sum2 + Y  =>  X - Y = d.
        # Also X >= z1, Y >= z2. Setting X = Y + d, we get Y >= z2 and Y >= z1 - d.
        # Thus Y = max(z2, z1 - d). The minimum final sum is sum2 + Y.
        d = sum2 - sum1
        Y = max(z2, z1 - d)
        # X would be Y + d, but we only need the final sum = sum2 + Y.
        return sum2 + Y