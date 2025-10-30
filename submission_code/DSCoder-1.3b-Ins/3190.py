class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        n, m = len(nums1), len(nums2)
        if n * m > 2 * sum(nums1) or n * m < 2 * sum(nums2):
            return -1
        if n > m:
            nums1, nums2 = nums2, nums1
            n, m = m, n
        nums1.sort()
        nums2.sort()
        i, j, operations = 0, 0, 0
        while i < n:
            while j < m and nums1[i] > nums2[j]:
                j += 1
            if j == m:
                return -1
            operations += (min(nums1[i] - nums2[j], nums2[j] - nums1[i]) + 1)
            i += 1
            j += 1
        return operations