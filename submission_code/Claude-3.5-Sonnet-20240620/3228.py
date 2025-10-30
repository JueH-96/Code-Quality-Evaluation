class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        half = n // 2
        
        set1 = set(nums1)
        set2 = set(nums2)
        
        unique1 = len(set1)
        unique2 = len(set2)
        
        # Count elements that are in both sets
        common = len(set1.intersection(set2))
        
        # Calculate how many unique elements we can take from each set
        from1 = min(unique1, half)
        from2 = min(unique2, half)
        
        # Calculate the total unique elements we can have
        total = from1 + from2
        
        # If the total exceeds n, we need to adjust
        if total > n:
            total = n
        
        return total