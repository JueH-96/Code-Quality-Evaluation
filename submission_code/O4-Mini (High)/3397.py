class Solution:
    def addedInteger(self, nums1: List[int], nums2: List[int]) -> int:
        # Sort both lists so the corresponding elements reflect the same shifts.
        nums1.sort()
        nums2.sort()
        # Since nums2[i] == nums1[i] + x for all i, x is the difference at any index.
        return nums2[0] - nums1[0]