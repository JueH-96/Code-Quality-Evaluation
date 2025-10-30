from typing import List

class Solution:
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        sum1 = sum(x for x in nums1 if x != 0)
        count1 = nums1.count(0)
        sum2 = sum(x for x in nums2 if x != 0)
        count2 = nums2.count(0)
        
        s1 = sum1 + count1
        s2 = sum2 + count2
        
        if s1 == s2:
            return s1
        elif s1 > s2:
            if count2 == 0:
                return -1
            else:
                return s1
        else:
            if count1 == 0:
                return -1
            else:
                return s2