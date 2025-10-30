class Solution:
    def findIntersectionValues(self, nums1: List[int], nums2: List[int]) -> List[int]:
        count1 = 0
        count2 = 0
        set1 = set(nums1)
        set2 = set(nums2)
        for i in range(len(nums1)):
            if nums1[i] in set2:
                count1 += 1
        for i in range(len(nums2)):
            if nums2[i] in set1:
                count2 += 1
        return [count1, count2]