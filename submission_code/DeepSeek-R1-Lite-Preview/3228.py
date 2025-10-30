class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        half = n // 2
        
        set1 = set(nums1)
        set2 = set(nums2)
        
        unique1 = set1 - set2
        unique2 = set2 - set1
        common = set1 & set2
        
        # Select unique1 elements from nums1, up to n/2
        unique1_selected = min(len(unique1), half)
        # Select unique2 elements from nums2, up to n/2
        unique2_selected = min(len(unique2), half)
        
        # Calculate remaining slots to fill with common elements
        remaining_from_nums1 = half - unique1_selected
        remaining_from_nums2 = half - unique2_selected
        total_remaining_slots = remaining_from_nums1 + remaining_from_nums2
        
        # Add common elements to the set, up to the number of common elements available
        common_selected = min(total_remaining_slots, len(common))
        
        # Total unique elements in the set s
        set_size = unique1_selected + unique2_selected + common_selected
        
        return set_size