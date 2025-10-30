class Solution:
    def findIntersectionValues(self, nums1: List[int], nums2: List[int]) -> List[int]:
        set1 = set(nums1)
        set2 = set(nums2)
        intersection1 = [value for value in nums1 if value in set2]
        intersection2 = [value for value in nums2 if value in set1]
        return [len(set(intersection1)), len(set(intersection2))]