class Solution:
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        sum1 = sum(nums1)
        c1 = nums1.count(0)
        sum2 = sum(nums2)
        c2 = nums2.count(0)
        
        if c1 == 0 and c2 == 0:
            return sum1 if sum1 == sum2 else -1
        elif c1 == 0:
            if sum1 >= sum2 + c2:
                return sum1
            else:
                return -1
        elif c2 == 0:
            if sum2 >= sum1 + c1:
                return sum2
            else:
                return -1
        else:
            s1_min = sum1 + c1
            s2_min = sum2 + c2
            return max(s1_min, s2_min)