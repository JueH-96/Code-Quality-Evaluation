class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        keep = n // 2
        
        set1 = set(nums1)
        set2 = set(nums2)
        
        only1 = set1 - set2  # Elements only in nums1
        only2 = set2 - set1  # Elements only in nums2
        both = set1 & set2   # Elements in both arrays
        
        # From nums1, prioritize keeping elements only in nums1
        keep1_unique = min(len(only1), keep)
        remaining1 = keep - keep1_unique
        
        # From nums2, prioritize keeping elements only in nums2
        keep2_unique = min(len(only2), keep)
        remaining2 = keep - keep2_unique
        
        # Use remaining slots for common elements
        keep_both = min(len(both), remaining1 + remaining2)
        
        return keep1_unique + keep2_unique + keep_both