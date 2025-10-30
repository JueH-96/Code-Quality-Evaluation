class Solution:
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        z1 = sum(1 for x in nums1 if x == 0)
        z2 = sum(1 for x in nums2 if x == 0)
        s1 = sum(nums1)
        s2 = sum(nums2)
        
        if s1 + z1 == s2 + z2:
            return s1 + z1
        if s1 + z1 < s2 + z2:
            return -1 if z2 == 0 else s2 + z2
        return -1 if z1 == 0 else s1 + z1