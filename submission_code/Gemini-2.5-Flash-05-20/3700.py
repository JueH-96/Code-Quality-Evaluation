import collections
from typing import List

class Solution:
    def subsequencesWithMiddleMode(self, nums: List[int]) -> int:
        n = len(nums)
        MOD = 10**9 + 7
        
        ans = 0
        
        # Helper function to calculate combinations C(n, 2)
        def combinations2(count):
            if count < 2:
                return 0
            return (count * (count - 1) // 2) % MOD

        # left_freq: Counter for elements in nums[0...k-1]
        left_freq = collections.Counter()
        # right_freq: Counter for elements in nums[k+1...n-1]
        # Initially, it contains all elements. Elements are removed as k progresses.
        right_freq = collections.Counter(nums)

        # Iterate through each possible index k for the middle element (seq[2])
        for k in range(n):
            val = nums[k]
            
            # Before calculating for current k:
            # 1. Remove nums[k] from right_freq. It's now the middle element.
            right_freq[val] -= 1
            if right_freq[val] == 0:
                del right_freq[val]

            # Only proceed if there are enough elements to form a subsequence of size 5:
            # Need at least 2 elements before k (indices i0, i1) and 2 elements after k (indices i3, i4).
            # This means k must be at least 2 (for i0=0, i1=1)
            # And k must be at most n-3 (for i3=k+1, i4=k+2, max index is n-1, so k+2 <= n-1 => k <= n-3)
            if k >= 2 and k <= n - 3:
                # --- Calculate statistics for the left side (elements nums[0]...nums[k-1]) ---
                L_count_val = left_freq[val]
                # Total count of elements in nums[0...k-1] that are not `val`
                L_total_other = (k - L_count_val)
                
                L_ways_vv = combinations2(L_count_val) # Ways to pick (val, val) from left
                L_ways_vo = (L_count_val * L_total_other) % MOD # Ways to pick (val, other) from left
                
                # Ways to pick two 'other' elements (not `val`) from left
                L_ways_oo_total = combinations2(L_total_other) # Total pairs of indices of 'other' elements
                L_ways_oo_same = 0 # Ways to pick (x, x) where x != val (same value chosen twice)
                for x_val, count_x_val in left_freq.items():
                    if x_val != val:
                        L_ways_oo_same = (L_ways_oo_same + combinations2(count_x_val)) % MOD
                # Ways to pick (x, y) where x != y and x,y != val
                L_ways_oo_distinct = (L_ways_oo_total - L_ways_oo_same + MOD) % MOD

                # --- Calculate statistics for the right side (elements nums[k+1]...nums[n-1]) ---
                R_count_val = right_freq[val]
                # Total count of elements in nums[k+1...n-1] that are not `val`
                R_total_other = ((n - 1 - k) - R_count_val) 
                
                R_ways_vv = combinations2(R_count_val) # Ways to pick (val, val) from right
                R_ways_vo = (R_count_val * R_total_other) % MOD # Ways to pick (val, other) from right
                
                R_ways_oo_total = combinations2(R_total_other)
                R_ways_oo_same = 0
                for x_val, count_x_val in right_freq.items():
                    if x_val != val:
                        R_ways_oo_same = (R_ways_oo_same + combinations2(count_x_val)) % MOD
                R_ways_oo_distinct = (R_ways_oo_total - R_ways_oo_same + MOD) % MOD

                # --- Combine counts based on the required frequency pattern of `val` ---
                # A subsequence seq = [l1, l2, val, r1, r2].
                # val must be the unique mode, meaning val's frequency F_val >= 2,
                # and for any x != val in seq, freq(x) == 1.

                # Case 1: val appears 5 times -> [v, v, v, v, v]
                #   Requires picking (val, val) from left and (val, val) from right.
                ans = (ans + (L_ways_vv * R_ways_vv) % MOD) % MOD

                # Case 2: val appears 4 times -> [v, v, v, v, x] (x != v)
                #   Requires picking (val, val) from one side and (val, other_single) from the other.
                #   A) L:(v,v), R:(v,x)
                ans = (ans + (L_ways_vv * R_ways_vo) % MOD) % MOD
                #   B) L:(v,x), R:(v,v)
                ans = (ans + (L_ways_vo * R_ways_vv) % MOD) % MOD

                # Case 3: val appears 3 times -> [v, v, v, x, y] (x != y, x,y != v)
                #   Requires:
                #   A) L:(v,v), R:(x,y) (x,y distinct and !=v)
                ans = (ans + (L_ways_vv * R_ways_oo_distinct) % MOD) % MOD
                #   B) L:(x,y) (x,y distinct and !=v), R:(v,v)
                ans = (ans + (L_ways_oo_distinct * R_ways_vv) % MOD) % MOD
                #   C) L:(v,x), R:(v,y) where x != y (x,y distinct and !=v)
                #      Total ways to pick (v,x) from L and (v,y) from R is L_ways_vo * R_ways_vo.
                #      We must subtract cases where x == y.
                sum_prod_vo_same_other_val = 0
                for x_val in left_freq: # Iterate distinct elements in left_freq (includes current `val` which is filtered below)
                    # `right_freq[x_val]` will correctly be 0 if `x_val` is not present in `right_freq`
                    if x_val != val:
                        ways_L_v_x = (L_count_val * left_freq[x_val]) % MOD
                        ways_R_v_x = (R_count_val * right_freq[x_val]) % MOD
                        sum_prod_vo_same_other_val = (sum_prod_vo_same_other_val + (ways_L_v_x * ways_R_v_x) % MOD) % MOD
                ans = (ans + L_ways_vo * R_ways_vo - sum_prod_vo_same_other_val + MOD) % MOD

                # Case 4: val appears 2 times -> [v, v, x, y, z] (x,y,z distinct, x,y,z != v)
                #   Requires one `val` from middle, one `val` from L or R, and 3 distinct elements (x,y,z) != `val`.
                #   A) L:(v,x), R:(y,z) where x,y,z are distinct and != v.
                #      Iterate through each specific `x_val` (other than `val`) that can be chosen from left.
                term_Lvo_Rod = 0
                for x_val in left_freq:
                    if x_val != val:
                        ways_val_x_L = (L_count_val * left_freq[x_val]) % MOD
                        
                        # Calculate R_ways_oo_distinct excluding x_val
                        # `cur_R_total_other` is elements in R (excluding `val` and `x_val`).
                        cur_R_total_other = R_total_other - right_freq[x_val]
                        cur_R_ways_oo_total = combinations2(cur_R_total_other)
                        
                        # `cur_R_ways_oo_same` is `R_ways_oo_same` but without pairs of `x_val`.
                        cur_R_ways_oo_same = (R_ways_oo_same - combinations2(right_freq[x_val]) + MOD) % MOD
                        
                        cur_R_ways_oo_distinct = (cur_R_ways_oo_total - cur_R_ways_oo_same + MOD) % MOD
                        
                        term_Lvo_Rod = (term_Lvo_Rod + (ways_val_x_L * cur_R_ways_oo_distinct) % MOD) % MOD
                ans = (ans + term_Lvo_Rod) % MOD

                #   B) Symmetric: L:(y,z), R:(v,x) where x,y,z are distinct and != v.
                term_Rvo_Lod = 0
                for x_val in right_freq:
                    if x_val != val:
                        ways_val_x_R = (R_count_val * right_freq[x_val]) % MOD
                        
                        # Calculate L_ways_oo_distinct excluding x_val
                        cur_L_total_other = L_total_other - left_freq[x_val]
                        cur_L_ways_oo_total = combinations2(cur_L_total_other)
                        
                        cur_L_ways_oo_same = (L_ways_oo_same - combinations2(left_freq[x_val]) + MOD) % MOD
                        
                        cur_L_ways_oo_distinct = (cur_L_ways_oo_total - cur_L_ways_oo_same + MOD) % MOD
                        
                        term_Rvo_Lod = (term_Rvo_Lod + (ways_val_x_R * cur_L_ways_oo_distinct) % MOD) % MOD
                ans = (ans + term_Rvo_Lod) % MOD
            
            # After calculations for current k, add nums[k] to left_freq.
            # This prepares left_freq for the next iteration (k+1), where nums[k] will be part of the left side.
            left_freq[val] += 1
            
        return ans