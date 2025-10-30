class Solution:
    def addedInteger(self, nums1: List[int], nums2: List[int]) -> int:
        # Since all elements in nums1 are increased by the same integer x
        # to become nums2, we can simply subtract the first element in
        # nums1 from the first element in nums2 to find x.
        return nums2[0] - nums1[0]