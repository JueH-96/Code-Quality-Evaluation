class Solution:
    def addedInteger(self, nums1: List[int], nums2: List[int]) -> int:
        # Sort both arrays
        nums1_sorted = sorted(nums1)
        nums2_sorted = sorted(nums2)

        # The difference between the first elements in the sorted arrays
        # will be the integer x that was added to each element of nums1.
        x = nums2_sorted[0] - nums1_sorted[0]

        return x