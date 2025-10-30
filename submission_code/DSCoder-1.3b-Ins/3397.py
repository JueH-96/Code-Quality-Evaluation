class Solution:
    def addedInteger(self, nums1: List[int], nums2: List[int]) -> int:
        for i in range(len(nums1)):
            if nums1[i] != nums2[i]:
                return nums1[i] - nums2[i]
        return 0