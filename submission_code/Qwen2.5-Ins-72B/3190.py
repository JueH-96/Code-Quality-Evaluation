from typing import List

class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        max1, max2 = nums1[-1], nums2[-1]
        
        # Function to count the number of operations needed to make the last element the maximum
        def count_ops(arr1, arr2, max_val):
            count = 0
            for i in range(n - 1):
                if arr1[i] > max_val and arr2[i] > max_val:
                    return float('inf')
                if arr1[i] > max_val:
                    count += 1
            return count
        
        # Count operations without swapping the last elements
        ops1 = count_ops(nums1, nums2, max1)
        ops2 = count_ops(nums2, nums1, max2)
        
        # Count operations with swapping the last elements
        nums1[-1], nums2[-1] = nums2[-1], nums1[-1]
        ops1_swap = count_ops(nums1, nums2, max2) + 1
        ops2_swap = count_ops(nums2, nums1, max1) + 1
        
        # Find the minimum number of operations
        min_ops = min(ops1, ops2, ops1_swap, ops2_swap)
        
        return min_ops if min_ops != float('inf') else -1