from typing import List

class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        max1, max2 = nums1[-1], nums2[-1]
        no_swap = 0
        swap = 1
        
        for i in range(n - 1):
            if nums1[i] <= max1 and nums2[i] <= max2:
                continue
            elif nums1[i] <= max2 and nums2[i] <= max1:
                swap += 1
            elif nums1[i] <= max1 or nums2[i] <= max2:
                no_swap += 1
            else:
                return -1
        
        return min(no_swap, swap)