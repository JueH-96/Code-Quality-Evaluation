class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        unique1 = len(set(nums1))
        unique2 = len(set(nums2))
        common = len(set(nums1) & set(nums2))
        
        max_remove1 = min(n // 2, unique1)
        max_remove2 = min(n // 2, unique2)
        
        remaining1 = unique1 - max_remove1
        remaining2 = unique2 - max_remove2
        
        if common <= n // 2:
            return n
        else:
            return min(remaining1 + remaining2 + (n // 2 - common), n)