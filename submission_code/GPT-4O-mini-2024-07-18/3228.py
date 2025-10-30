class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        from collections import Counter
        
        count1 = Counter(nums1)
        count2 = Counter(nums2)
        
        # Total elements to keep from both arrays
        to_keep = len(nums1) // 2
        
        # Unique elements in both arrays
        unique1 = set(count1.keys())
        unique2 = set(count2.keys())
        
        # Combine unique elements
        combined_unique = unique1.union(unique2)
        
        # Maximum size of the set we can achieve
        return min(len(combined_unique), to_keep + len(combined_unique) - to_keep)