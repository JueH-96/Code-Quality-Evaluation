class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        n, m = len(nums1), len(nums2)
        if n > m * 6 or m > n * 6:
            return -1
        sum1, sum2 = sum(nums1), sum(nums2)
        if sum1 > sum2:
            sum1, sum2 = sum2, sum1
            nums1, nums2 = nums2, nums1
        nums1.sort()
        nums2.sort(reverse=True)
        i, j, operations = 0, 0, 0
        while sum1 < sum2:
            if j >= m or (i < n and 6 - nums1[i] > nums2[j] - 1):
                sum1 += 6 - nums1[i]
                i += 1
            else:
                sum2 -= nums2[j] - 1
                j += 1
            operations += 1
        return operations