class Solution:
    def addedInteger(self, nums1: List[int], nums2: List[int]) -> int:
        # Since nums1[i] + x = nums2[i], we can rearrange to find x = nums2[i] - nums1[i]
        # We can use any pair of corresponding elements to find x, as it should be consistent
        # across all pairs due to the problem constraints.
        return nums2[0] - nums1[0]