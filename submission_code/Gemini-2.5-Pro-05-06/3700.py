from collections import Counter
from typing import List

class Solution:
    def subsequencesWithMiddleMode(self, nums: List[int]) -> int:
        N = len(nums)
        MOD = 10**9 + 7

        ans = 0

        def combinations2(n: int) -> int:
            if n < 2:
                return 0
            # n * (n - 1) is always even, so integer division is fine.
            return n * (n - 1) // 2

        for j in range(N):
            M = nums[j]
            
            # Calculate counts for the left part nums[0...j-1]
            L_M_count = 0  # Count of M on the left
            L_O_count = 0  # Count of non-M elements on the left
            L_O_freq_map = Counter() # Frequencies of non-M values on the left
            for k in range(j):
                if nums[k] == M:
                    L_M_count += 1
                else:
                    L_O_count += 1
                    L_O_freq_map[nums[k]] += 1
            
            # Calculate counts for the right part nums[j+1...N-1]
            R_M_count = 0  # Count of M on the right
            R_O_count = 0  # Count of non-M elements on the right
            R_O_freq_map = Counter() # Frequencies of non-M values on the right
            for k in range(j + 1, N):
                if nums[k] == M:
                    R_M_count += 1
                else:
                    R_O_count += 1
                    R_O_freq_map[nums[k]] += 1

            current_ans_for_j = 0

            # Scenario 1: 5 M's. (M, M, M, M, M)
            # Requires 2 M's from left, 2 M's from right.
            term1 = combinations2(L_M_count) * combinations2(R_M_count)
            current_ans_for_j = (current_ans_for_j + term1) % MOD

            # Scenario 2: 4 M's, 1 V (V != M). (M, M, M, M, V)
            # Case 2a: V from left. (1 M, 1 O from left; 2 M's from right)
            term2_V_left = (L_M_count * L_O_count) * combinations2(R_M_count)
            # Case 2b: V from right. (2 M's from left; 1 M, 1 O from right)
            term2_V_right = combinations2(L_M_count) * (R_M_count * R_O_count)
            current_ans_for_j = (current_ans_for_j + term2_V_left + term2_V_right) % MOD
            
            # Scenario 3: 3 M's, 2 non-M's V,W (V,W != M; V can be same as W). (M, M, M, V, W)
            # Case 3a: V,W from left. (2 O's from left; 2 M's from right)
            term3_VW_left = combinations2(L_O_count) * combinations2(R_M_count)
            # Case 3b: V from left, W from right. (1 M, 1 O from left; 1 M, 1 O from right)
            term3_Vleft_Wright = (L_M_count * L_O_count) * (R_M_count * R_O_count)
            # Case 3c: V,W from right. (2 M's from left; 2 O's from right)
            term3_VW_right = combinations2(L_M_count) * combinations2(R_O_count)
            current_ans_for_j = (current_ans_for_j + term3_VW_left + term3_Vleft_Wright + term3_VW_right) % MOD

            # Scenario 4: 2 M's, 3 non-M's V,W,X (V,W,X distinct values, != M). (M, M, V, W, X)
            # One M is nums[j]. The other M is from left or right.
            # The three non-M's V,W,X are split between left and right.
            
            # Precompute sum of squares of frequencies for non-M values.
            # Used for calculating ways to pick two distinct non-M values.
            L_O_sum_val_sq = 0
            for val_count in L_O_freq_map.values():
                L_O_sum_val_sq += val_count * val_count
            
            R_O_sum_val_sq = 0
            for val_count in R_O_freq_map.values():
                R_O_sum_val_sq += val_count * val_count

            # Case 4a: Other M from left. One non-M (V_L) from left. Two non-M (W,X) from right.
            #   (s0,s1) are (M, V_L) or (V_L, M). (s3,s4) are (W,X).
            #   V_L, W, X must be distinct values.
            term4_1_sum_over_V_L = 0
            if L_M_count > 0: # If we can pick an M from left
                for val_L, count_val_L_left in L_O_freq_map.items():
                    # count_val_L_left: ways to pick val_L as V_L from left.
                    
                    # Effective sum of counts and sum of squares of counts for right side,
                    # excluding val_L (if present on right side).
                    S_eff_R = R_O_count - R_O_freq_map.get(val_L, 0)
                    S_sq_eff_R = R_O_sum_val_sq - (R_O_freq_map.get(val_L, 0) ** 2)
                    
                    ways_WX_right = 0
                    if S_eff_R >= 2: # Need at least 2 elements to pick W,X from
                        # Ways to pick two positions with distinct values:
                        ways_WX_right = (S_eff_R * S_eff_R - S_sq_eff_R) // 2
                    
                    term4_1_sum_over_V_L = (term4_1_sum_over_V_L + count_val_L_left * ways_WX_right) % MOD
            
            term4_1 = (L_M_count * term4_1_sum_over_V_L) % MOD

            # Case 4b: Other M from right. One non-M (V_R) from right. Two non-M (W,X) from left.
            #   (s0,s1) are (W,X). (s3,s4) are (M, V_R) or (V_R, M).
            #   V_R, W, X must be distinct values.
            term4_2_sum_over_V_R = 0
            if R_M_count > 0: # If we can pick an M from right
                for val_R, count_val_R_right in R_O_freq_map.items():
                    # count_val_R_right: ways to pick val_R as V_R from right.

                    S_eff_L = L_O_count - L_O_freq_map.get(val_R, 0)
                    S_sq_eff_L = L_O_sum_val_sq - (L_O_freq_map.get(val_R, 0) ** 2)

                    ways_WX_left = 0
                    if S_eff_L >= 2:
                        ways_WX_left = (S_eff_L * S_eff_L - S_sq_eff_L) // 2
                    
                    term4_2_sum_over_V_R = (term4_2_sum_over_V_R + count_val_R_right * ways_WX_left) % MOD
            
            term4_2 = (R_M_count * term4_2_sum_over_V_R) % MOD
            
            current_ans_for_j = (current_ans_for_j + term4_1 + term4_2) % MOD
            
            ans = (ans + current_ans_for_j) % MOD
            
        return ans