class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        set1 = set(nums1)
        set2 = set(nums2)
        common = set1.intersection(set2)
        unique1 = set1 - common
        unique2 = set2 - common
        
        # Calculate the maximum number of unique elements we can take from each set
        max_unique1 = min(len(unique1), n // 2)
        max_unique2 = min(len(unique2), n // 2)
        
        # Calculate the remaining slots for common elements
        remaining1 = n // 2 - max_unique1
        remaining2 = n // 2 - max_unique2
        
        # Calculate the maximum number of common elements we can take
        max_common = min(len(common), remaining1 + remaining2)
        
        # The maximum size of the set is the sum of unique and common elements
        return max_unique1 + max_unique2 + max_common