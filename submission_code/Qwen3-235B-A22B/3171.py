from typing import List

class Solution:
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        # Calculate sum of non-zero elements and count zeros for nums1
        sum1_non_zero = sum(x for x in nums1 if x != 0)
        z1 = nums1.count(0)
        
        # Calculate sum of non-zero elements and count zeros for nums2
        sum2_non_zero = sum(x for x in nums2 if x != 0)
        z2 = nums2.count(0)
        
        # Case when both arrays have no zeros
        if z1 == 0 and z2 == 0:
            return sum1_non_zero if sum1_non_zero == sum2_non_zero else -1
        
        # Case when nums1 has no zeros (sum1 is fixed)
        elif z1 == 0:
            required_s2_min = sum2_non_zero + z2
            if sum1_non_zero >= required_s2_min:
                return sum1_non_zero
            else:
                return -1
        
        # Case when nums2 has no zeros (sum2 is fixed)
        elif z2 == 0:
            required_s1_min = sum1_non_zero + z1
            if sum2_non_zero >= required_s1_min:
                return sum2_non_zero
            else:
                return -1
        
        # Both arrays have at least one zero, can adjust to reach max of both min sums
        else:
            s1_min = sum1_non_zero + z1
            s2_min = sum2_non_zero + z2
            return max(s1_min, s2_min)