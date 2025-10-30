class Solution:
    def minOperations(self, nums1: list[int], nums2: list[int]) -> int:
        max1 = max(nums1)
        max2 = max(nums2)
        index1 = nums1.index(max1)
        index2 = nums2.index(max2)
        if max1 > nums2[-1] and max2 > nums1[-1]:
            nums1[index1], nums2[index1] = nums2[index1], nums1[index1]
            nums1[index2], nums2[index2] = nums2[index2], nums1[index2]
            return 2
        elif max1 > nums2[-1]:
            nums1[index1], nums2[index1] = nums2[index1], nums1[index1]
            return 1
        elif max2 > nums1[-1]:
            nums1[index2], nums2[index2] = nums2[index2], nums1[index2]
            return 1
        else:
            return -1