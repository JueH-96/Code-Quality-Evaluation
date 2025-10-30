class Solution:
    def addedInteger(self, nums1: List[int], nums2: List[int]) -> int:
        # Since nums1 becomes equal to nums2 by adding x to each element,
        # we can find x by subtracting the sum of nums1 from the sum of nums2
        return sum(nums2) - sum(nums1)