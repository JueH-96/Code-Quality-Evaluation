import math
from typing import List
from collections import Counter

class Solution:
    def subsequencesWithMiddleMode(self, nums: List[int]) -> int:
        MOD = 1000000007
        n = len(nums)
        ans = 0
        for mid_idx in range(2, n - 2):  # mid_idx from 2 to n-3 inclusive
            M = nums[mid_idx]
            # Left part
            left_nums = nums[:mid_idx]
            left_M_count = sum(1 for x in left_nums if x == M)
            nonM_left_count = len(left_nums) - left_M_count
            freq_left_nonM = Counter(x for x in left_nums if x != M)
            same_val_pairs_left = sum((freq * (freq - 1) // 2) for freq in freq_left_nonM.values())
            
            # Right part
            right_nums = nums[mid_idx + 1:]
            right_M_count = sum(1 for x in right_nums if x == M)
            nonM_right_count = len(right_nums) - right_M_count
            freq_right_nonM = Counter(x for x in right_nums if x != M)
            same_val_pairs_right = sum((freq * (freq - 1) // 2) for freq in freq_right_nonM.values())
            
            # Compute num_left_A[a] for a=0,1,2
            num_left_A = [
                (nonM_left_count * (nonM_left_count - 1) // 2),  # a=0
                (left_M_count * nonM_left_count),                # a=1
                (left_M_count * (left_M_count - 1) // 2)         # a=2
            ]
            
            # Compute num_right_B[b] for b=0,1,2
            num_right_B = [
                (nonM_right_count * (nonM_right_count - 1) // 2),  # b=0
                (right_M_count * nonM_right_count),                 # b=1
                (right_M_count * (right_M_count - 1) // 2)          # b=2
            ]
            
            # Compute num_ways_AB_ge2
            num_ways_AB_ge2 = 0
            for a in range(3):
                for b in range(3):
                    if a + b >= 2:
                        prod_ab = (num_left_A[a] * num_right_B[b]) % MOD
                        num_ways_AB_ge2 = (num_ways_AB_ge2 + prod_ab) % MOD
            
            # Compute sum_A1B0_distinct
            sum_A1B0_distinct = 0
            for V in freq_left_nonM:
                freq_left_V = freq_left_nonM[V]
                freq_right_V = freq_right_nonM.get(V, 0)
                num_S_X = nonM_right_count - freq_right_V
                comb_S_X_2 = max(0, (num_S_X * (num_S_X - 1) // 2))  # Ensure non-negative
                comb_freq_right_V_2 = max(0, (freq_right_V * (freq_right_V - 1) // 2))
                num_good_right_pairs = max(0, comb_S_X_2 - same_val_pairs_right + comb_freq_right_V_2)  # Ensure non-negative
                num_left_pairs_with_X_V = (freq_left_V * left_M_count)
                prod = ((num_left_pairs_with_X_V * num_good_right_pairs) % MOD)
                sum_A1B0_distinct = (sum_A1B0_distinct + prod) % MOD
            
            # Compute sum_A0B1_distinct
            sum_A0B1_distinct = 0
            for W in freq_right_nonM:
                freq_right_W = freq_right_nonM[W]
                freq_left_W = freq_left_nonM.get(W, 0)
                num_T_W = nonM_left_count - freq_left_W
                comb_T_W_2 = max(0, (num_T_W * (num_T_W - 1) // 2))  # Ensure non-negative
                comb_freq_left_W_2 = max(0, (freq_left_W * (freq_left_W - 1) // 2))
                num_good_left_pairs = max(0, comb_T_W_2 - same_val_pairs_left + comb_freq_left_W_2)  # Ensure non-negative
                num_right_pairs_with_R_W = (freq_right_W * right_M_count)
                prod = ((num_right_pairs_with_R_W * num_good_left_pairs) % MOD)
                sum_A0B1_distinct = (sum_A0B1_distinct + prod) % MOD
            
            # Total for this mid_idx
            total_for_mid = (num_ways_AB_ge2 + sum_A1B0_distinct + sum_A0B1_distinct) % MOD
            ans = (ans + total_for_mid) % MOD
        
        return ans