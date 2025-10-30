class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        max1, max2 = max(nums1), max(nums2)
        if max1 != nums1[-1] or max2 != nums2[-1]:
            if max1 != max2:
                return -1
            else:
                return min(sum(x < max1 for x in nums1), sum(x < max2 for x in nums2))
        else:
            return 0