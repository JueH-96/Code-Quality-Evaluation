import math

class Solution:
    def minCostGoodCaption(self, caption: str) -> str:
        n = len(caption)

        # dp_costs[i][char_code][run_len] stores the minimum cost
        # to transform caption[0...i] such that caption[i] becomes chr(char_code + ord('a'))
        # and it's part of a run of length run_len.
        # run_len can be 1, 2, or 3 (where 3 means 3 or more).
        dp_costs = [[[math.inf] * 4 for _ in range(26)] for _ in range(n)]

        # dp_prev_states[i][char_code][run_len] stores the (prev_char_code, prev_run_len)
        # that led to this minimum cost for path reconstruction.
        # Using None initially indicates no valid path found yet.
        dp_prev_states = [[[None] * 4 for _ in range(26)] for _ in range(n)]

        # Base case for i = 0 (first character of the caption)
        original_char_code_0 = ord(caption[0]) - ord('a')
        for target_char_code in range(26):
            cost_char = abs(original_char_code_0 - target_char_code)
            dp_costs[0][target_char_code][1] = cost_char
            dp_prev_states[0][target_char_code][1] = (-1, -1) # Sentinel for the start of the path

        # Fill DP table for i from 1 to n-1
        for i in range(1, n):
            original_char_code_i = ord(caption[i]) - ord('a')
            
            # Iterate target_char_code from 'a' to 'z' (0 to 25)
            # This ensures that if multiple characters lead to the same minimum cost at position i,
            # we automatically prefer the lexicographically smallest one.
            for target_char_code in range(26): 
                cost_current_char = abs(original_char_code_i - target_char_code)

                # --- Transition 1: current_run_len = 1 (start a new run) ---
                # This means the character at i-1 must have been different from target_char_code,
                # and its run must have been of length 3 (or more).
                min_prev_cost_l1 = math.inf
                best_prev_char_code_l1 = -1
                
                # Iterate prev_char_code from 'a' to 'z' (0 to 25)
                # This ensures that if multiple previous characters lead to the same min_prev_cost_l1,
                # we prefer the lexicographically smallest prev_char_code.
                for prev_char_code in range(26):
                    if prev_char_code == target_char_code: # New run must start with a different character
                        continue
                    
                    if dp_costs[i-1][prev_char_code][3] < min_prev_cost_l1:
                        min_prev_cost_l1 = dp_costs[i-1][prev_char_code][3]
                        best_prev_char_code_l1 = prev_char_code
                
                if min_prev_cost_l1 != math.inf:
                    new_cost = min_prev_cost_l1 + cost_current_char
                    # If new_cost is strictly less, update.
                    # If new_cost is equal, we don't update because the current state
                    # (target_char_code, 1) would have been reached by a path that
                    # already ensured lexicographical minimality up to i-1.
                    if new_cost < dp_costs[i][target_char_code][1]:
                        dp_costs[i][target_char_code][1] = new_cost
                        dp_prev_states[i][target_char_code][1] = (best_prev_char_code_l1, 3)

                # --- Transition 2: current_run_len = 2 (continue run of length 1) ---
                # This means the character at i-1 must have been the same (target_char_code),
                # and its run must have been of length 1.
                if dp_costs[i-1][target_char_code][1] != math.inf:
                    new_cost = dp_costs[i-1][target_char_code][1] + cost_current_char
                    if new_cost < dp_costs[i][target_char_code][2]:
                        dp_costs[i][target_char_code][2] = new_cost
                        dp_prev_states[i][target_char_code][2] = (target_char_code, 1) # Previous char was also target_char_code, run_len was 1

                # --- Transition 3: current_run_len = 3 (continue run of length 2 or 3) ---
                # This means the character at i-1 must have been the same (target_char_code),
                # and its run must have been of length 2 or 3.
                min_prev_cost_l3 = math.inf
                best_prev_run_len_l3 = -1
                
                # Check option from previous run_len 2
                if dp_costs[i-1][target_char_code][2] != math.inf:
                    min_prev_cost_l3 = dp_costs[i-1][target_char_code][2]
                    best_prev_run_len_l3 = 2
                
                # Check option from previous run_len 3
                # If both run_len 2 and run_len 3 lead to the same cost, we prefer run_len 2.
                # This is because (char, 2) is a "newer" run segment than (char, 3), and choosing it
                # implies we are forming the block of 3 from its earliest possible point, which
                # contributes to lexicographical preference.
                if dp_costs[i-1][target_char_code][3] < min_prev_cost_l3:
                    min_prev_cost_l3 = dp_costs[i-1][target_char_code][3]
                    best_prev_run_len_l3 = 3
                
                if min_prev_cost_l3 != math.inf:
                    new_cost = min_prev_cost_l3 + cost_current_char
                    if new_cost < dp_costs[i][target_char_code][3]:
                        dp_costs[i][target_char_code][3] = new_cost
                        dp_prev_states[i][target_char_code][3] = (target_char_code, best_prev_run_len_l3)

        # Find the overall minimum cost among all valid ending states at index n-1
        min_final_cost = math.inf
        final_char_code = -1
        final_run_len = -1 # This will always be 3 for a valid good caption

        # Iterate char_code from 'a' to 'z' (0 to 25) to ensure lexicographical smallest final character
        for char_code in range(26):
            # A good caption must end with a run of length at least 3, so we check run_len=3.
            if dp_costs[n-1][char_code][3] < min_final_cost:
                min_final_cost = dp_costs[n-1][char_code][3]
                final_char_code = char_code
                final_run_len = 3
            elif dp_costs[n-1][char_code][3] == min_final_cost:
                # If costs are equal, the earlier char_code (already processed) would have been chosen.
                # No need for explicit update here, as loop naturally handles lexicographical order.
                pass 

        # If no valid good caption could be formed (min_final_cost is still infinity)
        if min_final_cost == math.inf:
            return ""

        # Reconstruct the path (the resulting string)
        result = [''] * n
        current_char_code = final_char_code
        current_run_len = final_run_len

        # Backtrack from n-1 down to 0
        for i in range(n - 1, -1, -1):
            result[i] = chr(current_char_code + ord('a'))
            prev_char_code, prev_run_len = dp_prev_states[i][current_char_code][current_run_len]
            current_char_code = prev_char_code
            current_run_len = prev_run_len
        
        return "".join(result)