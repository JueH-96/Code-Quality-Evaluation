import collections
from typing import List

class Solution:
    def maxSubarrays(self, n: int, conflictingPairs: List[List[int]]) -> int:
        
        # Step 1: Preprocess conflictingPairs to group by L and find base_min_R and second_min_R
        # base_min_R_arr[L]: minimum R for pairs starting at L
        # second_min_R_arr[L]: second minimum R for pairs starting at L
        # We need N+1 size arrays as values are 1-indexed up to N
        base_min_R_arr = [n + 1] * (n + 1)
        second_min_R_arr = [n + 1] * (n + 1)
        
        # Use a list of lists to store all R values for each L to easily find min and second min
        pos_R_map = collections.defaultdict(list)
        
        for a, b in conflictingPairs:
            L, R = min(a, b), max(a, b)
            pos_R_map[L].append(R)
        
        for L in range(1, n + 1):
            if pos_R_map[L]:
                pos_R_map[L].sort() # Sort to easily pick min and second min
                base_min_R_arr[L] = pos_R_map[L][0]
                if len(pos_R_map[L]) >= 2:
                    second_min_R_arr[L] = pos_R_map[L][1]
        
        # Step 2: Calculate initial suffix_min_R array
        # suffix_min_R[i]: min R_k such that L_k >= i, considering base_min_R_arr[i] and suffix_min_R[i+1]
        suffix_min_R = [n + 1] * (n + 2) # n+1 for 1-indexing, n+2 for suffix_min_R[n+1] base case
        
        for i in range(n, 0, -1):
            suffix_min_R[i] = min(base_min_R_arr[i], suffix_min_R[i+1])
            
        # Step 3: Calculate initial total count of valid subarrays
        initial_total_valid_subarrays = 0
        for i in range(1, n + 1):
            initial_total_valid_subarrays += max(0, suffix_min_R[i] - i)
            
        max_valid_subarrays = initial_total_valid_subarrays
        
        # Step 4: Iterate through each pair to remove and calculate new total count
        # and find the maximum
        
        for a_j, b_j in conflictingPairs:
            L_j, R_j = min(a_j, b_j), max(a_j, b_j)
            
            # Start with the current total count if this pair were not removed.
            # This sum will be adjusted based on changes to suffix_min_R.
            current_valid_subarrays_temp = initial_total_valid_subarrays
            
            # Determine the effective new base_min_R for L_j after removing [L_j, R_j]
            # If R_j was the smallest R value for L_j, its replacement is the second smallest.
            # Otherwise, base_min_R_arr[L_j] remains unchanged.
            effective_min_R_at_L_j = base_min_R_arr[L_j] 
            if base_min_R_arr[L_j] == R_j:
                effective_min_R_at_L_j = second_min_R_arr[L_j]
            
            # Propagate changes from L_j downwards to 1.
            # 'prev_new_s_m_r' is effectively the new suffix_min_R[i+1] when calculating suffix_min_R[i]
            prev_new_s_m_r = suffix_min_R[L_j + 1] # For i=L_j, this is the original suffix_min_R[L_j+1]
            
            # Calculate the new suffix_min_R for L_j
            current_s_m_r = min(effective_min_R_at_L_j, prev_new_s_m_r)

            # If suffix_min_R[L_j] does not change after removal, no further propagation to the left
            if current_s_m_r == suffix_min_R[L_j]:
                max_valid_subarrays = max(max_valid_subarrays, current_valid_subarrays_temp)
                continue

            # Update the sum for L_j itself
            current_valid_subarrays_temp += (max(0, current_s_m_r - L_j) - max(0, suffix_min_R[L_j] - L_j))
            
            # Propagate changes to the left (for i from L_j-1 down to 1)
            for i in range(L_j - 1, 0, -1):
                # The 'prev_new_s_m_r' for current 'i' is the 'current_s_m_r' calculated for 'i+1'
                prev_new_s_m_r = current_s_m_r
                
                # Calculate the new suffix_min_R for current i
                current_s_m_r = min(base_min_R_arr[i], prev_new_s_m_r)
                
                # If the new suffix_min_R[i] is the same as the original,
                # propagation stops, as elements further left are not affected.
                if current_s_m_r == suffix_min_R[i]:
                    break
                
                # Add the difference to the total valid subarrays
                current_valid_subarrays_temp += (max(0, current_s_m_r - i) - max(0, suffix_min_R[i] - i))
            
            max_valid_subarrays = max(max_valid_subarrays, current_valid_subarrays_temp)
            
        return max_valid_subarrays