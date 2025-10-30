import typing
from typing import List

class Solution:
    def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
        MOD = 1000000007
        sorted_req = sorted(requirements, key=lambda x: x[0])
        current_len = 1
        current_inv = 0
        current_num_ways = 1
        
        for end_i, cnt_i in sorted_req:
            target_len = end_i + 1
            target_inv_val = cnt_i
            delta_len = target_len - current_len
            if delta_len < 0:
                return 0  # Should not happen if sorted, but safety check
            elif delta_len == 0:
                if target_inv_val != current_inv:
                    return 0
                # No change, continue
            else:  # delta_len > 0
                delta_inv_needed = target_inv_val - current_inv
                if delta_inv_needed < 0:
                    return 0  # Inversion count decreasing, impossible
                num_seq = self.num_sequences(current_len, target_len, delta_inv_needed, MOD)
                if num_seq == 0:
                    return 0
                current_num_ways = (current_num_ways * num_seq) % MOD
                current_inv = target_inv_val
                current_len = target_len
        
        return current_num_ways % MOD
    
    def num_sequences(self, a: int, b: int, delta_inv: int, mod: int) -> int:
        delta_adds = b - a
        if delta_inv < 0 or delta_adds < 0:
            return 0
        if delta_adds == 0:
            return 1 if delta_inv == 0 else 0
        
        prev_f = [0] * (delta_inv + 1)
        prev_f[0] = 1
        
        for add_step in range(delta_adds):
            max_k = a + add_step
            new_f = [0] * (delta_inv + 1)
            
            # Compute prefix sum of prev_f
            prefix_sum = [0]
            cum = 0
            for val in prev_f:
                cum = (cum + val) % mod
                prefix_sum.append(cum)
            
            for s in range(delta_inv + 1):
                low = max(0, s - max_k)
                high = s
                sum_range = (prefix_sum[high + 1] - prefix_sum[low] + mod) % mod
                new_f[s] = sum_range
            
            prev_f = new_f
        
        return prev_f[delta_inv]