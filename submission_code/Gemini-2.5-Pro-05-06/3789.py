import math

class Solution:
    def maxSubarrays(self, n: int, conflictingPairs: list[list[int]]) -> int:
        if n == 0:
            return 0
        
        # Handle trivial case: if only one conflicting pair, removing it makes all subarrays valid.
        if len(conflictingPairs) == 1:
            return n * (n + 1) // 2

        # Store conflicting pairs: adj[l] contains list of r values for pairs (l,r)
        # Values u,v are 1-indexed, convert to 0-indexed l,r
        adj = [[] for _ in range(n)]
        for u, v in conflictingPairs:
            l, r = min(u, v) - 1, max(u, v) - 1 
            adj[l].append(r)

        # Sort r-values for each l to easily find 1st and 2nd minimum r
        for i in range(n):
            adj[i].sort()

        # Calculate min_r_val_orig[i]: for a starting index i, what's the smallest r_p for any pair (i, r_p)?
        # If no pair starts at i, this is n.
        min_r_val_orig = [n] * n
        for i in range(n):
            if adj[i]:
                min_r_val_orig[i] = adj[i][0]
        
        # Calculate suffix_min_r_orig[i]: R_boundary_for_i
        # This is min(min_r_val_orig[k] for k >= i).
        # Iterating i from n-1 down to 0: suffix_min_r_orig[i] = min(min_r_val_orig[i], suffix_min_r_orig[i+1])
        suffix_min_r_orig = [n] * (n + 1) # suffix_min_r_orig[n] = n as base for loop
        current_min_limit = n
        for i in range(n - 1, -1, -1):
            current_min_limit = min(current_min_limit, min_r_val_orig[i])
            suffix_min_r_orig[i] = current_min_limit
        
        # Calculate original total count of valid subarrays and contributions per starting index
        count_orig = 0
        contrib_orig = [0] * n 
        for i in range(n):
            # Number of valid subarrays starting at i is suffix_min_r_orig[i] - i
            val = suffix_min_r_orig[i] - i
            if val > 0:
                contrib_orig[i] = val
                count_orig += val
        
        # Precompute suffix sums of contributions:
        # orig_contrib_sum_from_idx_plus_1[k] = sum of contrib_orig[j] for j from k+1 to n-1
        orig_contrib_sum_from_idx_plus_1 = [0] * (n + 1) # Entry n is 0, used for base k=n-1
        for i in range(n - 1, -1, -1): # Loop i from n-1 down to 0
            if i + 1 < n: # contrib_orig[i+1] exists
                 orig_contrib_sum_from_idx_plus_1[i] = contrib_orig[i+1] + orig_contrib_sum_from_idx_plus_1[i+1]
            # else i=n-1 (or i+1=n), sum from n to n-1 is 0. Default value handles this.
        
        max_total_count = count_orig # Default if removed pair doesn't change anything vital

        # Iterate through each pair, consider removing it
        for pair_info in conflictingPairs:
            u_rem, v_rem = pair_info
            l_k, r_k = min(u_rem, v_rem) - 1, max(u_rem, v_rem) - 1 # 0-indexed pair to remove
            
            # Original minimum r for l_k
            actual_min_at_lk_orig = min_r_val_orig[l_k]

            new_min_at_lk = actual_min_at_lk_orig
            
            # If r_k is the one defining min_r_val_orig[l_k]
            if r_k == actual_min_at_lk_orig:
                if len(adj[l_k]) == 1: # r_k was the only r-value for this l_k
                    new_min_at_lk = n
                elif adj[l_k][1] > r_k: # r_k was unique minimum, next one adj[l_k][1] is larger
                    new_min_at_lk = adj[l_k][1]
                # else adj[l_k][1] == r_k, so min_r_val_orig[l_k] is still r_k (multiple copies). No change.
            
            if new_min_at_lk == actual_min_at_lk_orig:
                # No change in min_r_val_orig[l_k], or r_k wasn't the minimum.
                # So suffix_min_r_orig array doesn't change. Count is count_orig.
                # max_total_count is already considering this possibility through its initialization.
                pass
            else: 
                # min_r_val_orig[l_k] effectively increased. Recalculate sum for affected part (i <= l_k).
                # Sum of contributions for i > l_k remains unchanged.
                sum_gt_lk = orig_contrib_sum_from_idx_plus_1[l_k]
                
                sum_lte_lk_new = 0
                # This is min(min_r_val_orig[j] for j > l_k), which is suffix_min_r_orig[l_k+1]
                current_min_overall_recalc = suffix_min_r_orig[l_k+1] 
                
                for i in range(l_k, -1, -1): # Iterate i from l_k down to 0
                    # Get the min_r_val at index i for this recalculation
                    val_at_i_for_min_calc = min_r_val_orig[i]
                    if i == l_k: # For l_k itself, use the new increased minimum
                        val_at_i_for_min_calc = new_min_at_lk
                    
                    current_min_overall_recalc = min(current_min_overall_recalc, val_at_i_for_min_calc)
                    term = current_min_overall_recalc - i
                    if term > 0:
                        sum_lte_lk_new += term
                
                current_score = sum_lte_lk_new + sum_gt_lk
                max_total_count = max(max_total_count, current_score)
                
        return max_total_count