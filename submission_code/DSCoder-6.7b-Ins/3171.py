class Solution:
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        nums1.sort()
        nums2.sort()
        sum1, sum2 = sum(nums1), sum(nums2)
        i = j = 0
        while i < len(nums1) and j < len(nums2):
            if nums1[i] < nums2[j]:
                sum1 += nums2[j] - nums1[i]
                i += 1
            elif nums1[i] > nums2[j]:
                sum2 += nums1[i] - nums2[j]
                j += 1
            else:
                i += 1
                j += 1
        while i < len(nums1):
            sum1 += nums1[i]
            i += 1
        while j < len(nums2):
            sum2 += nums2[j]
            j += 1
        return min(sum1, sum2) if sum1 != sum2 else -1