class Solution:
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        sum1 = sum(x for x in nums1 if x != 0)
        count1 = nums1.count(0)
        sum2 = sum(x for x in nums2 if x != 0)
        count2 = nums2.count(0)
        
        if count1 == 0 and count2 == 0:
            if sum1 == sum2:
                return sum1
            else:
                return -1
        elif count1 == 0:
            if sum1 >= sum2 + count2:
                return sum1
            else:
                return -1
        elif count2 == 0:
            if sum2 >= sum1 + count1:
                return sum2
            else:
                return -1
        else:
            s1 = sum1 + count1
            s2 = sum2 + count2
            s_candidate = max(s1, s2)
            return s_candidate