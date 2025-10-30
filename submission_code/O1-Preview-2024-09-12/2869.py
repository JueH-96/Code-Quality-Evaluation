class Solution:
    def maxNonDecreasingLength(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        len1, len2 = 1, 1
        res = 1
        for i in range(1, n):
            temp_len1, temp_len2 = len1, len2
            len1 = len2 = 1
            if nums1[i] >= nums1[i - 1]:
                len1 = max(len1, temp_len1 + 1)
            if nums1[i] >= nums2[i - 1]:
                len1 = max(len1, temp_len2 + 1)
            if nums2[i] >= nums1[i - 1]:
                len2 = max(len2, temp_len1 + 1)
            if nums2[i] >= nums2[i - 1]:
                len2 = max(len2, temp_len2 + 1)
            res = max(res, len1, len2)
        return res