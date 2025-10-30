import sys

# Setting recursion depth is crucial for potential deep comparisons based on N
# Max N is 5*10^4. A typical recursion depth limit is 1000. This will likely be insufficient.
# Set a higher limit, e.g., 60000, which should cover N=50000 plus some buffer.
# Note: This might fail on platforms with strict limits or cause stack overflow if N is excessively large.
try:
    sys.setrecursionlimit(60000) 
except Exception as e:
    # If setting recursion limit fails, print a warning but continue.
    # The program might still work for smaller N or if deep recursion isn't triggered.
    print(f"Warning: Could not set recursion depth: {e}", file=sys.stderr)

class Solution:
    
    def compare_strings(self, idx1, idx2):
        """ Compares the reconstructed strings ending at indices idx1 and idx2 lexicographically.
            Uses memoization via self.compare_memo dictionary.
            Returns:
                -1 if string corresponding to idx1 is lexicographically smaller than string for idx2.
                 0 if strings are equal.
                 1 if string corresponding to idx1 is lexicographically larger than string for idx2.
        """
        # Base cases for comparison
        if idx1 == idx2: return 0
        # Empty string (idx=0) is smallest
        if idx1 == 0: return -1 
        if idx2 == 0: return 1

        # Use tuple for memoization key, ensure canonical order (smaller index first)
        # This maximizes cache hits by storing comparison result for pair (min(idx1, idx2), max(idx1, idx2))
        state = tuple(sorted((idx1, idx2))) 
        if state in self.compare_memo:
             # Retrieve memoized result
             res = self.compare_memo[state]
             # Adjust result based on original order (idx1 vs idx2) compared to canonical order
             return res if idx1 == state[0] else -res 

        # If not memoized, retrieve DP states for idx1 and idx2
        # dp state format: (cost, last_char_ord, last_block_length, prev_idx)
        cost1, c1_ord, k1, j1 = self.dp[idx1] 
        cost2, c2_ord, k2, j2 = self.dp[idx2]

        # Recursively compare the prefix strings ending at j1 and j2
        prefix_cmp = self.compare_strings(j1, j2)

        final_res = 0 # Initialize result assuming strings might be equal
        if prefix_cmp != 0:
            # If prefixes differ, the comparison result is determined by the prefixes
            final_res = prefix_cmp
        else: 
            # Prefixes T_{j1} and T_{j2} are equal. Compare the last blocks.
            # The last blocks are c1^k1 and c2^k2 respectively.
            if c1_ord < c2_ord: final_res = -1 # Character c1 is smaller
            elif c1_ord > c2_ord: final_res = 1 # Character c1 is larger
            else: # Characters are the same (c1_ord == c2_ord)
                  # Compare block lengths k1 and k2.
                  # If T_j + c^k1 vs T_j + c^k2: smaller k means lexicographically smaller string.
                  # E.g., "aaa" < "aaaa"
                  if k1 < k2: final_res = -1
                  elif k1 > k2: final_res = 1
                  else: final_res = 0 # Equal length, char, and prefix means strings are identical
        
        # Store the computed result in memoization cache using canonical order key
        self.compare_memo[state] = final_res if idx1 == state[0] else -final_res
        # Return result based on original comparison order (idx1 vs idx2)
        return final_res


    def compare_options(self, opt1, opt2, i):
        """ Compares the full strings that would result from choosing option1 vs option2 for state i.
            An option 'opt' is defined as (prev_idx, char_ord).
            Returns -1 if opt1 leads to smaller string, 1 if opt2 does, 0 if equal.
        """
        # Unpack options into previous index (j) and character ordinal (c_ord)
        j1, c1_ord = opt1
        j2, c2_ord = opt2
        
        # Calculate block lengths for each option
        k1 = i - j1
        k2 = i - j2

        # First, compare the base prefix strings T_{j1} and T_{j2} using recursive string comparison
        prefix_cmp = self.compare_strings(j1, j2)
        if prefix_cmp != 0:
            # If prefixes differ, that determines the comparison outcome
            return prefix_cmp

        # Prefixes are equal. Now compare the final blocks c1^k1 and c2^k2.
        if c1_ord < c2_ord: return -1 # Character c1 is smaller
        if c1_ord > c2_ord: return 1 # Character c1 is larger
        
        # Characters c1 and c2 are also equal. Compare block lengths k1 and k2.
        if k1 < k2: return -1 # String from opt1 (T_j + c^k1) is prefix of string from opt2 (T_j + c^k2) => smaller
        if k1 > k2: return 1 # String from opt2 (T_j + c^k2) is prefix of string from opt1 (T_j + c^k1) => larger
        return 0 # Options lead to identical strings
    

    def minCostGoodCaption(self, caption: str) -> str:
        n = len(caption)
        # A good caption requires blocks of length at least 3. If n < 3, impossible.
        if n < 3:
            return "" 

        # Convert caption characters to 0-based ordinals for easier calculation
        s_ords = [ord(c) - ord('a') for c in caption]

        # Precompute prefix sums of costs for changing characters to each possible character 'a' through 'z'.
        # P[c_ord][k] = sum_{p=0..k-1} | ord(S[p]) - c_ord |
        P = [[0] * (n + 1) for _ in range(26)]
        for c_ord in range(26):
            current_sum = 0
            for k in range(n):
                current_sum += abs(s_ords[k] - c_ord)
                P[c_ord][k + 1] = current_sum

        # Helper function to compute cost of changing segment S[j..i-1] to char c_ord
        def Cost(j, i, c_ord):
            # Cost = P_c[i] - P_c[j]
            if j >= i: return 0
            # Basic bounds check for safety, though logic should ensure valid indices
            if i > n or j < 0: return float('inf') 
            return P[c_ord][i] - P[c_ord][j]

        # Initialize DP table. dp[i] stores the best state for prefix of length i.
        # State format: (min_cost, last_char_ordinal, last_block_length, previous_state_index)
        self.dp = [(float('inf'), -1, 0, -1)] * (n + 1)
        # Base case: dp[0] represents empty prefix, cost is 0.
        self.dp[0] = (0, -1, 0, -1) 
        
        # Initialize memoization cache for string comparisons
        self.compare_memo = {}

        # minV_state[k][c] stores minimum value of (dp[j].cost - P_c[j]) found for j in [0..k],
        # along with the index 'best_j' that achieved this minimum.
        # Format: (min_value, best_j_idx)
        minV_state = [[(float('inf'), -1)] * 26 for _ in range(n + 1)]

        # Initialize minV_state for k=0 based on the base case dp[0]
        if self.dp[0][0] == 0: # Check if base case is reachable (cost is 0)
            for c_ord in range(26):
                # Calculate V0 = dp[0].cost - P_c[0]. P_c[0] = 0.
                V0 = self.dp[0][0] - P[c_ord][0] # V0 will be 0
                minV_state[0][c_ord] = (V0, 0) # The best index for j=0 is 0 itself.

        # Main DP loop: compute dp[i] for i from 1 to n
        for i in range(1, n + 1):
            # Initialize the best state found so far for index i
            current_best_state_for_i = (float('inf'), -1, 0, -1)
            
            # Consider transitions only if a block of length >= 3 is possible ending at i
            if i >= 3: 
                # The previous state 'j' must end at index j-1, where j <= i-3.
                # We need the minimum value tracked up to index i-3.
                relevant_j_max_idx = i - 3 
                for c_ord in range(26): # Iterate through all possible characters for the last block
                    # Fetch the best state (min_val, best_j) ending at or before relevant_j_max_idx for char c_ord
                    min_val, best_j = minV_state[relevant_j_max_idx][c_ord]
                    
                    # Check if a valid predecessor state 'best_j' was found (best_j != -1)
                    if best_j != -1: 
                         # Additionally ensure the predecessor state dp[best_j] is actually reachable (cost is not infinity)
                         if self.dp[best_j][0] == float('inf'): continue 

                         # Calculate the cost of transitioning from state best_j to i using character c_ord
                         candidate_cost = self.dp[best_j][0] + Cost(best_j, i, c_ord)
                         
                         # Compare this candidate transition with the current best found for state i
                         if candidate_cost < current_best_state_for_i[0]:
                             # Found a strictly lower cost, update the best state for i
                             current_best_state_for_i = (candidate_cost, c_ord, i - best_j, best_j)
                         elif candidate_cost == current_best_state_for_i[0]:
                             # Costs are equal, need to tie-break using lexicographical order
                             candidate_option = (best_j, c_ord)
                             current_best_option = (current_best_state_for_i[3], current_best_state_for_i[1])
                             # Compare the strings resulting from these two options
                             cmp_res = self.compare_options(candidate_option, current_best_option, i)
                             if cmp_res == -1: # Candidate option leads to a lexicographically smaller string
                                 current_best_state_for_i = (candidate_cost, c_ord, i - best_j, best_j)
            
            # Store the determined best state for index i in the DP table
            self.dp[i] = current_best_state_for_i

            # Update minV_state[i] using the newly computed dp[i] state info
            # This info will be used for calculating future DP states (e.g., dp[i+3], dp[i+4]...)
            if self.dp[i][0] != float('inf'): # Only update if dp[i] is reachable
                 for c_ord in range(26):
                     # Calculate the value Vi = dp[i].cost - P_c[i] associated with state i for char c_ord
                     Vi = self.dp[i][0] - P[c_ord][i] 
                     
                     # Get the previous minimum state from minV_state[i-1] to compare against
                     prev_min_val, prev_best_j = (float('inf'), -1) # Default if i=0 (handled by initialization)
                     if i > 0: # Access state from i-1 if available (i>0)
                         prev_min_val, prev_best_j = minV_state[i-1][c_ord]
                     
                     # Compare the new value Vi with the previous minimum value prev_min_val
                     if Vi < prev_min_val:
                         # New value is strictly better, update minV state for index i
                         minV_state[i][c_ord] = (Vi, i)
                     elif Vi == prev_min_val:
                         # Values are equal, tie-break based on lexicographical comparison of corresponding strings
                         # Compare string ending at i (T_i) vs string ending at prev_best_j (T_{prev_best_j})
                         cmp_res = self.compare_strings(i, prev_best_j)
                         if cmp_res == -1: # String T_i is lexicographically smaller
                             minV_state[i][c_ord] = (Vi, i)
                         else: # Keep previous state T_{prev_best_j} as it's smaller or equal
                             minV_state[i][c_ord] = (prev_min_val, prev_best_j)
                     else: # Vi > prev_min_val, new value is worse, keep previous state
                         minV_state[i][c_ord] = (prev_min_val, prev_best_j)
            else: # dp[i] is unreachable, simply copy the state from minV_state[i-1]
                  # This propagates the infinity cost / unreachable status correctly
                if i > 0: 
                   for c_ord in range(26):
                        minV_state[i][c_ord] = minV_state[i-1][c_ord]
                # else: for i=0, minV_state[0] is already initialized correctly
        
        # After filling the DP table, reconstruct the final string from dp[n]
        if self.dp[n][0] == float('inf'):
            # If dp[n] cost is infinity, it means no good caption could be formed.
            return "" 

        # Reconstruct the string by backtracking from state n using prev_idx pointers
        res_parts = []
        curr_idx = n
        while curr_idx > 0:
            cost, c_ord, k, j = self.dp[curr_idx]
            # Basic sanity check: block length k should be positive if state is reachable
            if k <= 0: 
                 # This indicates an error state or logic flaw, should not happen for reachable final state.
                 print(f"Error: Encountered state with k={k} during reconstruction at index {curr_idx}", file=sys.stderr)
                 return "" # Return empty string indicating failure/error
            
            # Append the character block corresponding to this state to the result list
            res_parts.append(chr(ord('a') + c_ord) * k)
            # Move to the previous state index
            curr_idx = j
        
        # The reconstructed parts are in reverse order of blocks. Reverse the list and join.
        return "".join(res_parts[::-1])