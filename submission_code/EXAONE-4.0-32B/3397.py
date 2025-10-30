class Solution:
    def addedInteger(self, nums1: List[int], nums2: List[int]) -> int:
        total1 = sum(nums1)
        total2 = sum(nums2)
        n = len(nums1)
        return (total2 - total1) // n