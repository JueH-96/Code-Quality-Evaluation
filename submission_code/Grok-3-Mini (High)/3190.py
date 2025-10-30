from typing import List
import math

class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        L = n - 1
        ans = float('inf')
        
        def compute_cost_sum(V1, V2):
            cost_sum = 0
            for j in range(n):
                if j != L:
                    A_j = nums1[j]
                    B_j = nums2[j]
                    if (A_j <= V1 and B_j <= V2):  # no swap works
                        cost_j = 0
                    elif (B_j <= V1 and A_j <= V2):  # swap works
                        cost_j = 1
                    else:
                        return None  # impossible
                    cost_sum += cost_j
            return cost_sum  # possible, return the sum
        
        # Case 1: no swap at L
        V1_no = nums1[L]
        V2_no = nums2[L]
        cost_indices_no = compute_cost_sum(V1_no, V2_no)
        if cost_indices_no is not None:
            total_cost_no = cost_indices_no + 0  # no swap at L
            ans = min(ans, total_cost_no)
        
        # Case 2: swap at L
        V1_swap = nums2[L]
        V2_swap = nums1[L]
        cost_indices_swap = compute_cost_sum(V1_swap, V2_swap)
        if cost_indices_swap is not None:
            total_cost_swap = cost_indices_swap + 1  # swap at L
            ans = min(ans, total_cost_swap)
        
        if ans == float('inf'):
            return -1
        else:
            return ans