from typing import List

class Solution:
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        sum1 = sum(num for num in nums1 if num != 0)
        zeros1 = nums1.count(0)
        sum2 = sum(num for num in nums2 if num != 0)
        zeros2 = nums2.count(0)
        
        if zeros1 > 0 and zeros2 > 0:
            s1_min = sum1 + zeros1
            s2_min = sum2 + zeros2
            return max(s1_min, s2_min)
        else:
            if zeros1 == 0 and zeros2 == 0:
                return sum1 if sum1 == sum2 else -1
            elif zeros1 == 0:
                required = sum1
                min_sum2 = sum2 + zeros2
                if required >= min_sum2:
                    return required
                else:
                    return -1
            else:  # zeros2 == 0
                required = sum2
                min_sum1 = sum1 + zeros1
                if required >= min_sum1:
                    return required
                else:
                    return -1