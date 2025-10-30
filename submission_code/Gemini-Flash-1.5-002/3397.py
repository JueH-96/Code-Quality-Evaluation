class Solution:
    def addedInteger(self, nums1: List[int], nums2: List[int]) -> int:
        if not nums1 or not nums2 or len(nums1) != len(nums2):
            raise ValueError("Invalid input arrays")
        return nums2[0] - nums1[0]