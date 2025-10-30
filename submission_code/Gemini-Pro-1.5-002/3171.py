class Solution:
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        s1 = sum(nums1)
        s2 = sum(nums2)
        c1 = nums1.count(0)
        c2 = nums2.count(0)

        if s1 > s2:
            s1, s2 = s2, s1
            c1, c2 = c2, c1

        if c1 == 0 and s1 != s2:
            return -1
        
        if s1 == s2 and c1 == c2 == 0:
            return s1
        
        diff = s2 - s1
        
        if diff == 0:
            return s1 + c1 if c1 > 0 else s1

        if c1 > c2 and diff > 0:
            if diff <= c1 - c2:
                return s2
            else:
                return -1
        elif c1 < c2 and diff > 0:
            return s2
        elif c1 == c2 and diff > 0:
            return s2
        elif diff == 0:
            return s1 + c1 if c1 > 0 else s1
        else:
            return -1