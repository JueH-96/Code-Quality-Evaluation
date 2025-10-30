from collections import defaultdict
from typing import List

class Solution:
    def minChanges(self, nums: List[int], k: int) -> int:
        n = len(nums)
        n_pairs = n // 2
        
        cnt = defaultdict(int)
        freq_T = [0] * (k + 2)  # Covers 0 to k inclusive
        cnt_groups = defaultdict(list)
        
        for i in range(n_pairs):
            a = nums[i]
            b = nums[n - 1 - i]
            d = abs(a - b)
            cnt[d] += 1
            
            # Compute T_p for this pair
            T = max(a, b, k - a, k - b)
            if T <= k:
                freq_T[T] += 1
            cnt_groups[d].append(T)
        
        # Build suffix sum array ssum
        ssum = [0] * (k + 2)  # ssum[k+1] = 0
        for X in range(k, -1, -1):
            ssum[X] = ssum[X + 1] + freq_T[X]
        
        # Build sum_cond_d array
        sum_cond_d = [0] * (k + 1)
        for d in cnt_groups:
            for T in cnt_groups[d]:
                if T >= d:
                    sum_cond_d[d] += 1
        
        # Calculate the maximum sum_savings
        max_value = 0
        for X in range(k + 1):
            current_cnt = cnt.get(X, 0)
            current_ssum = ssum[X]
            current_sum_cond_d = sum_cond_d[X]
            value = 2 * current_cnt + (current_ssum - current_sum_cond_d)
            if value > max_value:
                max_value = value
        
        minimal_total_cost = 2 * n_pairs - max_value
        return minimal_total_cost