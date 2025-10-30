from typing import List

class Solution:
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        sum1 = sum(nums1)
        zeros1 = nums1.count(0)
        min_sum1 = sum1 + zeros1
        flex1 = (zeros1 > 0)
        
        sum2 = sum(nums2)
        zeros2 = nums2.count(0)
        min_sum2 = sum2 + zeros2
        flex2 = (zeros2 > 0)
        
        if flex1 and flex2:
            return max(min_sum1, min_sum2)
        elif not flex1 and not flex2:
            if min_sum1 == min_sum2:
                return min_sum1
            else:
                return -1
        elif flex1 and not flex2:
            if min_sum2 >= min_sum1:
                return min_sum2
            else:
                return -1
        elif not flex1 and flex2:
            if min_sum1 >= min_sum2:
                return min_sum1
            else:
                return -1