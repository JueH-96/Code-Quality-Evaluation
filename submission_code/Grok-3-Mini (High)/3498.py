import math
from typing import List

class Solution:
    def minChanges(self, nums: List[int], k: int) -> int:
        n = len(nums)
        total_pairs = n // 2
        
        # Initialize frequency arrays
        freq_M = [0] * (k + 1)
        freq_d = [0] * (k + 1)
        H = [0] * (k + 1)  # H[d] for each d
        
        # Iterate over each pair and compute M and d
        for i in range(total_pairs):
            p = nums[i]
            q = nums[n - 1 - i]
            A_p = max(p, k - p)
            A_q = max(q, k - q)
            M_val = max(A_p, A_q)
            d_val = abs(p - q)
            
            # Increment frequencies
            freq_M[M_val] += 1
            freq_d[d_val] += 1
            if M_val >= d_val:
                H[d_val] += 1
        
        # Compute suffix sum for M
        suffix_sum_M = [0] * (k + 1)
        current_sum = 0
        for m in range(k, -1, -1):  # from k to 0 inclusive
            current_sum += freq_M[m]
            suffix_sum_M[m] = current_sum  # number of pairs with M >= m
        
        # Now iterate over all possible X and find the minimum sum of changes
        min_changes = float('inf')
        for X in range(k + 1):
            sum_cost_X = 2 * total_pairs - suffix_sum_M[X] - 2 * freq_d[X] + H[X]
            if sum_cost_X < min_changes:
                min_changes = sum_cost_X
        
        return int(min_changes)