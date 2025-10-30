class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        if len(nums1) * len(nums2) == 0:
            return 0
        if len(nums1) * len(nums2) == 0:
            return -1
        sum1, sum2 = sum(nums1), sum(nums2)
        if sum1 == sum2:
            return 0
        if sum1 < sum2:
            nums1, nums2 = nums2, nums1
        nums1.sort()
        nums2.sort()
        i, j, ops = 0, len(nums2) - 1, 0
        while sum1 < sum2:
            if j < 0 or (i < len(nums1) and nums1[i] - nums2[j] > 0):
                sum1 += nums1[i] - nums2[j]
                i += 1
            else:
                sum2 -= nums2[j] - nums1[i]
                j -= 1
            ops += 1
        return ops