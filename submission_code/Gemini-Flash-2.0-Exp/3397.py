class Solution:
    def addedInteger(self, nums1: List[int], nums2: List[int]) -> int:
        x = nums2[0] - nums1[0]
        for i in range(len(nums1)):
            if nums1[i] + x != nums2[i]:
                return 0
        return x