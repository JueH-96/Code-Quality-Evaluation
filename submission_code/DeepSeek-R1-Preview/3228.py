class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        k = n // 2
        
        set1 = set(nums1)
        set2 = set(nums2)
        
        u1 = len(set1)
        u2 = len(set2)
        
        common = len(set1 & set2)
        
        max_a = min(u1, k)
        max_b = min(u2, k)
        
        total_unique = u1 + u2 - common
        sum_max = max_a + max_b
        
        return min(sum_max, total_unique)