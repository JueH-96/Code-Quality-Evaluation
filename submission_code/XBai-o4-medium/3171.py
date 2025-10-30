from typing import List

class Solution:
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        sum1 = 0
        c1 = 0
        for num in nums1:
            if num == 0:
                c1 += 1
            else:
                sum1 += num
        
        sum2 = 0
        c2 = 0
        for num in nums2:
            if num == 0:
                c2 += 1
            else:
                sum2 += num
        
        if c1 == 0 and c2 == 0:
            return sum1 if sum1 == sum2 else -1
        elif c1 == 0:
            if sum1 >= (sum2 + c2):
                return sum1
            else:
                return -1
        elif c2 == 0:
            if sum2 >= (sum1 + c1):
                return sum2
            else:
                return -1
        else:
            t1 = sum1 + c1
            t2 = sum2 + c2
            return max(t1, t2)