class Solution:
    def addedInteger(self, nums1: List[int], nums2: List[int]) -> int:
        # Sort both lists
        s1 = sorted(nums1)
        s2 = sorted(nums2)
        # Since it's given there's a valid integer x, we can compute it
        # by taking the difference between any corresponding pair.
        # We use the first pair for simplicity.
        return s2[0] - s1[0]