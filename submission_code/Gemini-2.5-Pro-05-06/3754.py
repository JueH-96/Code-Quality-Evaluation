import math

class Solution:
    def maxDistance(self, s: str, k: int) -> int:
        n = len(s)
        max_overall_dist = 0

        # Contributions of N, S, E, W to the four sums:
        # (X+Y), (X-Y), (-X+Y), (-X-Y)
        # N (0,1): [1, -1, 1, -1]
        # S (0,-1): [-1, 1, -1, 1]
        # E (1,0): [1, 1, -1, -1]
        # W (-1,0): [-1, -1, 1, 1]
        contrib_map = {
            'N': [1, -1, 1, -1],
            'S': [-1, 1, -1, 1],
            'E': [1, 1, -1, -1],
            'W': [-1, -1, 1, 1]
        }

        # current_sums_for_4_dirs[d] = sum of contributions for direction d using s[0...idx]
        current_sums_for_4_dirs = [0, 0, 0, 0]
        # num_negative_contrib_for_4_dirs[d] = count of chars in s[0...idx] contributing -1 to sum for dir d
        num_negative_contrib_for_4_dirs = [0, 0, 0, 0]

        def calculate_final_sum(initial_sum_val, num_negative_val, num_changes_to_use):
            # Number of "good" changes: convert a -1 contribution to +1.
            # Each such change increases sum by 2 and costs 1 operation.
            c_good = min(num_negative_val, num_changes_to_use)
            
            sum_after_good_changes = initial_sum_val + 2 * c_good
            
            # Changes remaining after making c_good changes.
            changes_remaining = num_changes_to_use - c_good
            
            final_sum_val = sum_after_good_changes
            # If changes_remaining is odd, we must make one more change.
            # This change will be applied to a character already contributing +1 (or changed to do so).
            # E.g. 'N' to 'S' for (X+Y) sum. Contribution changes from +1 to -1. Sum decreases by 2.
            if changes_remaining % 2 == 1:
                final_sum_val -= 2
            return final_sum_val

        for idx in range(n):
            # m is the number of moves, or current prefix length
            m = idx + 1 
            
            char = s[idx]
            char_contributions = contrib_map[char]
            
            for i in range(4): # For each of the 4 sum directions
                current_sums_for_4_dirs[i] += char_contributions[i]
                if char_contributions[i] == -1:
                    num_negative_contrib_for_4_dirs[i] += 1
            
            # Effective number of changes we can use for prefix of length m
            k_eff = min(k, m)

            for i in range(4): # For each direction
                # Calculate max possible sum using k_eff changes
                res_sum1 = calculate_final_sum(
                    current_sums_for_4_dirs[i],
                    num_negative_contrib_for_4_dirs[i],
                    k_eff
                )
                max_overall_dist = max(max_overall_dist, abs(res_sum1))
                
                # If k_eff >= 1, also calculate max possible sum using k_eff-1 changes
                # This handles the "at most k" allowing choice of parity for number of changes.
                if k_eff >= 1:
                    res_sum2 = calculate_final_sum(
                        current_sums_for_4_dirs[i],
                        num_negative_contrib_for_4_dirs[i],
                        k_eff - 1
                    )
                    max_overall_dist = max(max_overall_dist, abs(res_sum2))
                    
        return max_overall_dist