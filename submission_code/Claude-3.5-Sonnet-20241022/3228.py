class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        half = n // 2
        
        # Convert arrays to sets
        set1 = set(nums1)
        set2 = set(nums2)
        
        # Get unique elements in each array
        unique1 = len(set1)
        unique2 = len(set2)
        
        # We can take at most half elements from each array
        take1 = min(unique1, half)
        take2 = min(unique2, half)
        
        # Get union of both sets
        union = set1.union(set2)
        
        # Return min of total union and sum of elements we can take from each array
        return min(len(union), take1 + take2)