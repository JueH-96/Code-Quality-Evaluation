class Solution:
    def addedInteger(self, nums1: List[int], nums2: List[int]) -> int:
        # Since the question guarantees that there is an integer x such that adding x to every
        # element in nums1 will result in nums2, we can simply use the difference between the 
        # first elements.
        x = nums2[0] - nums1[0]
        return x