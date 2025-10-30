from typing import List

class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        max1, max2 = max(nums1), max(nums2)
        if nums1[n-1] == max1 and nums2[n-1] == max2:
            return 0
        if nums1[n-1] > max1 or nums2[n-1] > max2:
            return -1
        
        def helper(a, b):
            count = 0
            for i in range(n-1):
                if a[i] > a[n-1] and b[i] > a[n-1]:
                    return float('inf')
                if a[i] > a[n-1] or b[i] > a[n-1]:
                    count += 1
                    a[i], b[i] = b[i], a[i]
                    if a[i] > a[n-1] and b[i] > a[n-1]:
                        return float('inf')
            return count
        
        return min(helper(nums1[:], nums2[:]), 1 + helper(nums2[:], nums1[:]))