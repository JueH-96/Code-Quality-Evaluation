class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        set1 = set(nums1)
        set2 = set(nums2)
        
        # Elements unique to each array and common elements
        only_in_1 = set1 - set2
        only_in_2 = set2 - set1
        common = set1 & set2
        
        n = len(nums1)
        half = n // 2
        
        # We can keep at most half elements from each array
        # Priority: unique elements first, then common elements
        
        # From nums1: keep unique elements first, then common if space
        keep_from_1 = min(len(only_in_1), half)
        remaining_1 = half - keep_from_1
        common_from_1 = min(remaining_1, len(common))
        
        # From nums2: keep unique elements first, then common if space
        keep_from_2 = min(len(only_in_2), half)
        remaining_2 = half - keep_from_2
        common_from_2 = min(remaining_2, len(common))
        
        # Total unique elements we can get
        unique_elements = keep_from_1 + keep_from_2
        
        # For common elements, we don't double count
        # We can use at most all common elements
        common_elements = min(common_from_1 + common_from_2, len(common))
        
        return unique_elements + common_elements