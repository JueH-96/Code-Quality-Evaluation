class Solution:
    def findIntersectionValues(self, nums1: List[int], nums2: List[int]) -> List[int]:
        set1, set2 = set(nums1), set(nums2)
        nums1_inter, nums2_inter = 0, 0
        for num in nums1:
            if num in set2:
                nums1_inter += 1
        for num in nums2:
            if num in set1:
                nums2_inter += 1
        return [nums1_inter, nums2_inter]