class Solution:
    def addedInteger(self, nums1: List[int], nums2: List[int]) -> int:
        # Calculate the difference between the first elements of nums1 and nums2
        x = nums2[0] - nums1[0]
        return x