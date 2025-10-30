from typing import List

class Solution:
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        zeros1 = nums1.count(0)
        zeros2 = nums2.count(0)
        sum1 = sum(nums1) + zeros1
        sum2 = sum(nums2) + zeros2
        
        if zeros1 == 0 and sum1 < sum2 - zeros2:
            return -1
        if zeros2 == 0 and sum2 < sum1 - zeros1:
            return -1
        
        return max(sum1, sum2)