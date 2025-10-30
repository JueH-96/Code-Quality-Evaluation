from typing import List
from collections import defaultdict

class Solution:
    def subsequencesWithMiddleMode(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)

        def combinations(n, k):
            if k < 0 or k > n:
                return 0
            if k == 0 or k == n:
                return 1
            if k == 1 or k == n - 1:
                return n
            if k == 2 or k == n - 2:
                # n * (n-1) is always even, so integer division is fine.
                # Modulo after division.
                # For larger k, we would need modular inverse for factorials.
                # But for k=2, simple calculation is sufficient and avoids potential
                # large number issues before modulo if n is large but fits in standard integer types.
                # Python handles large integers, so (n * (n-1) // 2) % MOD is fine.
                return (n * (n - 1) // 2) % MOD
            return 0 # We only need k=0, 1, 2 for combinations of 2 elements

        total_ans = 0

        # countL stores frequencies of elements in nums[0...i-1]
        countL = defaultdict(int)
        # countR stores frequencies of elements in nums[i...n-1] initially
        countR = defaultdict(int)

        # Initial population of countR for i = 2 (contains counts for nums[2...n-1])
        for k in range(2, n):
            countR[nums[k]] += 1

        # Loop through possible middle indices i (nums[i])
        # The middle index i can be from 2 to n-3 to have at least 2 elements before and 2 elements after
        for i in range(2, n - 2):
            v = nums[i]

            # For index i, countR should represent frequencies in nums[i+1...n-1].
            # So, remove nums[i] from the current countR.
            countR[v] -= 1
            if countR[v] == 0:
                del countR[v]

            # Now countL is for nums[0..i-1], countR is for nums[i+1..n-1]

            cntL_v = countL[v]
            cntR_v = countR[v]
            
            # Number of indices j < i with nums[j] != v
            sum_countL_nv = i - cntL_v 
            # Number of indices k > i with nums[k] != v
            sum_countR_nv = (n - 1 - i) - cntR_v 

            # Number of ways to choose 2 indices from left part (0...i-1)
            # resulting in k_L values equal to v
            # N_L[k_L] = number of ways to choose 2 indices from nums[0...i-1]
            # such that exactly k_L of the chosen values are equal to v.
            N_L = [
                combinations(sum_countL_nv, 2), # k_L = 0: Choose 2 non-v from sum_countL_nv indices
                cntL_v * sum_countL_nv,       # k_L = 1: Choose 1 v from cntL_v indices, 1 non-v from sum_countL_nv indices
                combinations(cntL_v, 2)         # k_L = 2: Choose 2 v from cntL_v indices
            ]

            # Number of ways to choose 2 indices from right part (i+1...n-1)
            # resulting in k_R values equal to v
            N_R = [
                combinations(sum_countR_nv, 2), # k_R = 0: Choose 2 non-v from sum_countR_nv indices
                cntR_v * sum_countR_nv,       # k_R = 1: Choose 1 v from cntR_v indices, 1 non-v from sum_countR_nv indices
                combinations(cntR_v, 2)         # k_R = 2: Choose 2 v from cntR_v indices
            ]
            
            # Apply modulo to N_L and N_R counts
            N_L = [x % MOD for x in N_L]
            N_R = [x % MOD for x in N_R]

            # Case: freq(v) >= 3 in subsequence {nums[i1], nums[i2], v, nums[i4], nums[i5]}
            # This happens when k_L + k_R >= 2. In this case, v's frequency (>=3) is strictly greater
            # than the maximum possible frequency of any other element (at most 2 among the remaining 4).
            # So, v is always the unique mode.
            for k_L in range(3):
                for k_R in range(3):
                    if k_L + k_R >= 2:
                        total_ans = (total_ans + N_L[k_L] * N_R[k_R]) % MOD

            # Case: freq(v) = 2 in subsequence ({v, l_nv, r_nv1, r_nv2} or {l_nv1, l_nv2, v, r_nv})
            # This happens when k_L + k_R = 1.
            # v is the unique mode if and only if the 3 non-v elements are distinct.

            # Sum of C(count[x], 2) for all x != v in countL
            sum_C_countL_nv_2 = 0
            for x in countL:
                if x != v:
                    sum_C_countL_nv_2 = (sum_C_countL_nv_2 + combinations(countL[x], 2)) % MOD

            # Sum of C(count[x], 2) for all x != v in countR
            sum_C_countR_nv_2 = 0
            for x in countR:
                if x != v:
                    sum_C_countR_nv_2 = (sum_C_countR_nv_2 + combinations(countR[x], 2)) % MOD

            # Number of ways to choose 2 indices from L non-v indices s.t. values are distinct
            # Total pairs of indices from L non-v: C(sum_countL_nv, 2)
            # Subtract pairs of indices from L non-v where values are the same: sum_{x!=v} C(countL[x], 2)
            total_pairs_L_nv_distinct = (combinations(sum_countL_nv, 2) - sum_C_countL_nv_2 + MOD) % MOD

            # Number of ways to choose 2 indices from R non-v indices s.t. values are distinct
            total_pairs_R_nv_distinct = (combinations(sum_countR_nv, 2) - sum_C_countR_nv_2 + MOD) % MOD

            # Case k_L = 1, k_R = 0: Choose 1 v, 1 nv(x) from L; 2 nv(y,z) from R. Non-v: {x,y,z} distinct.
            # This means x!=v, y!=v, z!=v, x!=y, x!=z, y!=z.
            if cntL_v > 0: # Must be able to pick at least one v from L
                # Sum over distinct values x != v present in countL
                sum_waysL_1v1x_times_valid_waysR_2nv = 0
                for x in countL:
                    if x != v:
                        # Number of ways to choose 1 index < i with value v and 1 index < i with value x
                        waysL_1v1x = (cntL_v * countL[x]) % MOD

                        # Number of ways to choose 2 indices > i from non-v indices
                        # such that the values are distinct AND neither value is x.
                        # This is the number of ways to choose 2 indices from {k > i | nums[k] != v and nums[k] != x}
                        # such that their values are distinct.
                        
                        # Number of indices k > i with nums[k] != v and nums[k] != x
                        num_R_indices_not_v_not_x = sum_countR_nv - countR.get(x, 0)

                        # Total pairs of indices from {k > i | nums[k] != v and nums[k] != x}
                        total_pairs_R_not_v_not_x = combinations(num_R_indices_not_v_not_x, 2)

                        # Pairs of indices from {k > i | nums[k] != v and nums[k] != x} where values are same (must be y=y, y!=v, y!=x)
                        sum_C_countR_not_v_not_x_2 = 0
                        for y in countR:
                             if y != v and y != x:
                                 # countR[y] is the number of indices in R with value y.
                                 # These are the indices in {k > i | nums[k] != v and nums[k] != x} with value y.
                                 sum_C_countR_not_v_not_x_2 = (sum_C_countR_not_v_not_x_2 + combinations(countR[y], 2)) % MOD

                        # Number of ways to choose 2 indices from {k > i | nums[k] != v and nums[k] != x} with distinct values
                        ways_pick_2nv_R_distinct_not_x = (total_pairs_R_not_v_not_x - sum_C_countR_not_v_not_x_2 + MOD) % MOD

                        sum_waysL_1v1x_times_valid_waysR_2nv = (sum_waysL_1v1x_times_valid_waysR_2nv + waysL_1v1x * ways_pick_2nv_R_distinct_not_x) % MOD
                total_ans = (total_ans + sum_waysL_1v1x_times_valid_waysR_2nv) % MOD

            # Case k_L = 0, k_R = 1: Choose 2 nv(x,y) from L; 1 v, 1 nv(z) from R. Non-v: {x,y,z} distinct.
            # This means x!=v, y!=v, z!=v, x!=y, x!=z, y!=z.
            if cntR_v > 0: # Must be able to pick at least one v from R
                # Sum over distinct values z != v present in countR
                sum_waysR_1v1z_times_valid_waysL_2nv = 0
                for z in countR:
                     if z != v:
                        # Number of ways to choose 1 index > i with value v and 1 index > i with value z
                        waysR_1v1z = (cntR_v * countR[z]) % MOD

                        # Number of ways to choose 2 indices < i from non-v indices
                        # such that the values are distinct AND neither value is z.
                        num_L_indices_not_v_not_z = sum_countL_nv - countL.get(z, 0)
                        total_pairs_L_not_v_not_z = combinations(num_L_indices_not_v_not_z, 2)

                        sum_C_countL_not_v_not_z_2 = 0
                        for y in countL:
                            if y != v and y != z:
                                sum_C_countL_not_v_not_z_2 = (sum_C_countL_not_v_not_z_2 + combinations(countL[y], 2)) % MOD

                        ways_pick_2nv_L_distinct_not_z = (total_pairs_L_not_v_not_z - sum_C_countL_not_v_not_z_2 + MOD) % MOD
                        
                        sum_waysR_1v1z_times_valid_waysL_2nv = (sum_waysR_1v1z_times_valid_waysL_2nv + ways_pick_2nv_L_distinct_not_z * waysR_1v1z) % MOD
                total_ans = (total_ans + sum_waysR_1v1z_times_valid_waysL_2nv) % MOD


            # Update countL for next iteration (i -> i+1): add nums[i]
            countL[v] += 1
            
            # countR for next iteration (i+1) will cover nums[i+1...n-1].
            # The loop structure processes index i, updates countL and countR, then moves to i+1.
            # countL is correctly updated by adding nums[i].
            # countR is already updated at the start of the loop by removing nums[i].
            # For the next iteration (i+1), countR should represent nums[i+2...n-1].
            # The current countR represents nums[i+1...n-1].
            # So, at the start of the next iteration (i+1), we will remove nums[i+1] from countR.
            # This implies the initial countR outside the loop should be for nums[2...n-1].
            # And at the start of loop i, we remove nums[i].
            # This logic seems consistent with Algorithm v5.

        return total_ans