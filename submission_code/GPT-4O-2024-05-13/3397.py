class Solution:
    def addedInteger(self, nums1: List[int], nums2: List[int]) -> int:
        # Since nums1[i] + x = nums2[i], we can find x by subtracting nums1[i] from nums2[i]
        # We can use the first element to find x since all elements should have the same x
        return nums2[0] - nums1[0]