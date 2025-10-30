class Solution:
    def findIntersectionValues(self, nums1: List[int], nums2: List[int]) -> List[int]:
        nums2_set = set(nums2)
        nums1_set = set(nums1)
        count1 = sum(1 for x in nums1 if x in nums2_set)
        count2 = sum(1 for x in nums2 if x in nums1_set)
        return [count1, count2]