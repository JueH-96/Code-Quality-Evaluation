class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        set1 = set(nums1)
        set2 = set(nums2)
        
        u1 = set1 - set2
        u2 = set2 - set1
        c = set1 & set2
        
        len_u1 = len(u1)
        len_u2 = len(u2)
        len_c = len(c)
        
        half = n // 2
        
        min_u1 = min(len_u1, half)
        min_u2 = min(len_u2, half)
        
        remaining1 = half - min_u1
        remaining2 = half - min_u2
        
        common_take = min(len_c, remaining1 + remaining2)
        
        return min_u1 + min_u2 + common_take