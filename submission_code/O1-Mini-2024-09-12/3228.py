class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        set1 = set(nums1)
        set2 = set(nums2)
        
        unique1 = len(set1 - set2)
        unique2 = len(set2 - set1)
        shared = len(set1 & set2)
        
        n = len(nums1)
        half_n = n // 2
        
        allowed_keep_in_nums1 = min(unique1, half_n)
        allowed_keep_in_nums2 = min(unique2, half_n)
        
        remaining_slots_nums1 = half_n - allowed_keep_in_nums1
        remaining_slots_nums2 = half_n - allowed_keep_in_nums2
        
        shared_keep_count = min(shared, remaining_slots_nums1 + remaining_slots_nums2)
        
        max_s_size = allowed_keep_in_nums1 + allowed_keep_in_nums2 + shared_keep_count
        
        return max_s_size