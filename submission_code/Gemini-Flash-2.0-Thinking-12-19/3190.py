from typing import List

class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        min_ops = float('inf')
        
        # Case 1: last pair is (nums1[n-1], nums2[n-1])
        current_ops1 = 0
        possible1 = True
        for i in range(n - 1):
            condition_a = (nums1[i] <= nums1[n-1]) and (nums2[i] <= nums2[n-1])
            condition_b = (nums2[i] <= nums1[n-1]) and (nums1[i] <= nums2[n-1])
            if condition_a:
                pass # no operation needed
            elif condition_b:
                current_ops1 += 1
            else:
                possible1 = False
                break
        if possible1:
            min_ops = min(min_ops, current_ops1)
            
        # Case 2: last pair is (nums2[n-1], nums1[n-1]) - swap at index n-1
        current_ops2 = 1
        possible2 = True
        for i in range(n - 1):
            condition_c = (nums1[i] <= nums2[n-1]) and (nums2[i] <= nums1[n-1])
            condition_d = (nums2[i] <= nums2[n-1]) and (nums1[i] <= nums1[n-1])
            if condition_c:
                pass # no operation needed
            elif condition_d:
                current_ops2 += 1
            else:
                possible2 = False
                break
        if possible2:
            min_ops = min(min_ops, current_ops2)
            
        if min_ops == float('inf'):
            return -1
        else:
            return min_ops