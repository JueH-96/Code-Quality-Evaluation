class Solution:
    def addedInteger(self, nums1: List[int], nums2: List[int]) -> int:
        # Calculate the difference between corresponding elements
        # Since the arrays are of equal length, we can safely iterate
        # and compute the difference for each pair
        # The x should be the same for all pairs
        # So we can take the difference of the first pair as x
        x = nums2[0] - nums1[0]
        return x