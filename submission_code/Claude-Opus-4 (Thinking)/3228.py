class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        half = n // 2
        
        set1 = set(nums1)
        set2 = set(nums2)
        
        # Count unique elements in each category
        unique_to_1 = len(set1 - set2)
        unique_to_2 = len(set2 - set1)
        common = len(set1 & set2)
        
        # Greedily keep unique elements first
        keep_from_1 = min(unique_to_1, half)
        keep_from_2 = min(unique_to_2, half)
        
        # Use remaining capacity for common elements
        remaining_capacity_1 = half - keep_from_1
        remaining_capacity_2 = half - keep_from_2
        keep_common = min(common, remaining_capacity_1 + remaining_capacity_2)
        
        return keep_from_1 + keep_from_2 + keep_common