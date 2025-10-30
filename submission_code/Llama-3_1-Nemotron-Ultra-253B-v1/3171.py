from typing import List

class Solution:
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        sum1_fixed = sum(num for num in nums1 if num != 0)
        count1 = nums1.count(0)
        sum2_fixed = sum(num for num in nums2 if num != 0)
        count2 = nums2.count(0)
        
        s1_min = sum1_fixed + count1
        s2_min = sum2_fixed + count2
        T = max(s1_min, s2_min)
        
        valid = True
        
        # Check conditions for nums1
        if count1 == 0:
            if (T - sum1_fixed) != 0:
                valid = False
        else:
            if (T - sum1_fixed) < count1:
                valid = False
        
        # Check conditions for nums2
        if count2 == 0:
            if (T - sum2_fixed) != 0:
                valid = False
        else:
            if (T - sum2_fixed) < count2:
                valid = False
        
        return T if valid else -1