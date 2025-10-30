class Solution:
    def addedInteger(self, nums1: List[int], nums2: List[int]) -> int:
        sorted1 = sorted(nums1)
        sorted2 = sorted(nums2)
        x = sorted2[0] - sorted1[0]
        return x