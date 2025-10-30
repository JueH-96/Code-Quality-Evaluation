from typing import List

class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        n = len(nums1)
        initial_sum = sum(nums1)
        if initial_sum <= x:
            return 0
        sum_b = sum(nums2)
        sorted_pairs = sorted(zip(nums1, nums2), key=lambda p: (-p[1], -p[0]))
        
        for k in range(1, n + 1):
            a_sum = 0
            b_sum_terms = 0
            for i in range(k):
                a_i, b_i = sorted_pairs[i]
                a_sum += a_i
                b_sum_terms += b_i * (k - i)
            total_reduction = a_sum + b_sum_terms
            current_total = initial_sum + k * sum_b - total_reduction
            if current_total <= x:
                return k
        
        return -1