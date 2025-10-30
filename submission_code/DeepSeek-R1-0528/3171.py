class Solution:
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        s1 = sum(nums1)
        s2 = sum(nums2)
        z1 = nums1.count(0)
        z2 = nums2.count(0)
        
        base1 = s1 + z1
        base2 = s2 + z2
        
        if z1 == 0 and z2 == 0:
            return s1 if s1 == s2 else -1
        elif z1 == 0:
            if s1 < base2:
                return -1
            else:
                return s1
        elif z2 == 0:
            if s2 < base1:
                return -1
            else:
                return s2
        else:
            return max(base1, base2)