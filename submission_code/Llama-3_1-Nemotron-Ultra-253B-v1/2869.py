from typing import List

class Solution:
    def maxNonDecreasingLength(self, nums1: List[int], nums2: List[int]) -> int:
        if not nums1:
            return 0
        prev0 = 1  # choosing nums1[0]
        prev1 = 1  # choosing nums2[0]
        max_len = max(prev0, prev1)
        n = len(nums1)
        for i in range(1, n):
            a_prev = nums1[i-1]
            b_prev = nums2[i-1]
            a_curr = nums1[i]
            b_curr = nums2[i]
            
            current0 = 1
            if a_curr >= a_prev:
                current0 = max(current0, prev0 + 1)
            if a_curr >= b_prev:
                current0 = max(current0, prev1 + 1)
            
            current1 = 1
            if b_curr >= a_prev:
                current1 = max(current1, prev0 + 1)
            if b_curr >= b_prev:
                current1 = max(current1, prev1 + 1)
            
            prev0, prev1 = current0, current1
            max_len = max(max_len, current0, current1)
        
        return max_len