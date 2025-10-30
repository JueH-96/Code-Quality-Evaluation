class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        unique1 = set(nums1)
        unique2 = set(nums2)
        
        x = len(unique1 - unique2)  # Unique to nums1
        y = len(unique2 - unique1)  # Unique to nums2
        z = len(unique1 & unique2)  # Common elements
        
        x_kept = min(x, n // 2)
        y_kept = min(y, n // 2)
        
        available_z_slots = (n // 2 - x_kept) + (n // 2 - y_kept)
        t = min(z, available_z_slots)
        
        return x_kept + y_kept + t