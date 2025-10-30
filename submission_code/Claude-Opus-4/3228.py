class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        half = n // 2
        
        # Convert to sets to get unique elements
        set1 = set(nums1)
        set2 = set(nums2)
        
        # Find elements unique to each array and common elements
        unique1 = set1 - set2
        unique2 = set2 - set1
        common = set1 & set2
        
        # We can keep at most half elements from each array
        # Prioritize keeping unique elements from each array
        keep1 = min(len(unique1), half)
        keep2 = min(len(unique2), half)
        
        # Fill remaining slots with common elements
        remaining1 = half - keep1
        remaining2 = half - keep2
        
        # Common elements can be kept from either array
        # We want to maximize the total unique elements
        common_keep = min(len(common), remaining1 + remaining2)
        
        # Total unique elements = unique from array1 + unique from array2 + common
        return keep1 + keep2 + common_keep