class Solution:
    def addedInteger(self, nums1: List[int], nums2: List[int]) -> int:
        # Sort both arrays so that elements line up by frequency.
        nums1_sorted = sorted(nums1)
        nums2_sorted = sorted(nums2)
        
        # The difference between any corresponding pair (here the first pair)
        # is the integer x added to each element of nums1.
        return nums2_sorted[0] - nums1_sorted[0]