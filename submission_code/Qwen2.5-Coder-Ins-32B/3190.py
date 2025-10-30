from typing import List

class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        max1, max2 = nums1[-1], nums2[-1]
        
        def count_swaps(a, b):
            swaps = 0
            for i in range(n - 1):
                if a[i] > max1 or b[i] > max2:
                    return float('inf')
                if a[i] > max2 or b[i] > max1:
                    swaps += 1
            return swaps
        
        original = count_swaps(nums1, nums2)
        swapped = count_swaps(nums2, nums1) + 1
        
        result = min(original, swapped)
        return result if result != float('inf') else -1