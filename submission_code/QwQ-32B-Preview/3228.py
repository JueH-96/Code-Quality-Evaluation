class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        half = n // 2
        set1 = set(nums1)
        set2 = set(nums2)
        
        unique1 = set1 - set2
        unique2 = set2 - set1
        common = set1 & set2
        
        unique1_count = len(unique1)
        unique2_count = len(unique2)
        common_count = len(common)
        
        selected_unique1 = min(half, unique1_count)
        selected_unique2 = min(half, unique2_count)
        
        remaining_nums1 = half - selected_unique1
        remaining_nums2 = half - selected_unique2
        
        selected_common = min(remaining_nums1 + remaining_nums2, common_count)
        
        total_unique = selected_unique1 + selected_unique2 + selected_common
        
        return total_unique