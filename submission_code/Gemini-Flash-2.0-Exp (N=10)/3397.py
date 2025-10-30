class Solution:
    def addedInteger(self, nums1: List[int], nums2: List[int]) -> int:
        if not nums1:
            return 0
        return nums2[0] - nums1[0]