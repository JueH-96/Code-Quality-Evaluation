class Solution:
    def addedInteger(self, nums1: List[int], nums2: List[int]) -> int:
        # Calculate the difference between corresponding elements of nums2 and nums1
        differences = [nums2[i] - nums1[i] for i in range(len(nums1))]

        # Since all differences should be the same, we can return any of them
        return differences[0]