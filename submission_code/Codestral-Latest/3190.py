class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        max1 = max(nums1)
        max2 = max(nums2)

        # Check if it's possible to satisfy both conditions
        if nums1[n - 1] != max1 and nums2[n - 1] != max2:
            return -1

        # Find the index of the maximum value in nums1 and nums2
        idx1 = nums1.index(max1)
        idx2 = nums2.index(max2)

        # If both maximum values are at the last index, no operations are needed
        if idx1 == n - 1 and idx2 == n - 1:
            return 0

        # If one of the maximum values is at the last index, one operation is needed
        if idx1 == n - 1 or idx2 == n - 1:
            return 1

        # If both maximum values are not at the last index, two operations are needed
        return 2