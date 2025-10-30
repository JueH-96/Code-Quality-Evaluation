class Solution:
    def addedInteger(self, nums1: List[int], nums2: List[int]) -> int:
        # Sort both arrays
        nums1.sort()
        nums2.sort()
        
        # Compute the consistent difference
        return nums2[0] - nums1[0]