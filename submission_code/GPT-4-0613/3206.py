class Solution:
    def findIntersectionValues(self, nums1: List[int], nums2: List[int]) -> List[int]:
        nums1_set = set(nums1)
        nums2_set = set(nums2)
        intersection = nums1_set & nums2_set
        return [sum(1 for num in nums1 if num in intersection), sum(1 for num in nums2 if num in intersection)]