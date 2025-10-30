import collections

class Solution:
    def makeStringGood(self, s: str) -> int:
        n = len(s)
        
        # Convert s_counts to an array for easier indexing 0-25
        char_freq = [0] * 26
        s_counts = collections.Counter(s)
        for char_code in range(26):
            char = chr(ord('a') + char_code)
            char_freq[char_code] = s_counts[char]

        min_total_ops = float('inf')

        # Iterate over possible number of distinct characters in the good string (m)
        # m can range from 1 to 26
        for m in range(1, 27):
            # Iterate over possible frequency (k) for each distinct character
            # k can range from 0 (empty string) to n (all characters are the same)
            # The range for k can be slightly larger if insertions are cheap.
            # Maximum length of target string should not exceed n + 26*2 (max operations to change/insert all chars).
            # Max possible k: (n + 26*2) / m
            # A looser upper bound is `n + 26 * 25` if changes were linear, but they are bounded by 2.
            # So `n + 26 * 2` as a rough upper bound for length.
            # So k_max is roughly `n + 52`.
            # For simplicity, iterate up to `n + 52` (or just `n+1` since `n` itself is `2*10^4`).
            # The strict upper bound for `k` must be limited for performance.
            # If `m*k` becomes too large, `abs(n - m*k)` term dominates.
            # A safe upper bound for k can be `n + 2*26` (if all existing chars are deleted (cost N)
            # and N+52 new chars are inserted (cost N+52)).
            # The effective k range should be roughly `[0, n + 2 * 26]`.
            max_k = n + 2 * 26 # N + max (2*26) changes. Max 2 operations per change.
            if max_k > n + 1: # Just to limit it if n is small, ensures it doesn't try absurdly high k
                max_k = n + 1 
            
            for k in range(0, max_k + 1):
                target_length = m * k

                # OFFSET helps map balance values [-OFFSET, OFFSET] to [0, 2*OFFSET].
                # This OFFSET value is derived from the observation that any character
                # change beyond 1 step (e.g., 'a' to 'c') costs 2, which is the same as delete+insert.
                # So, only 'cost=1' moves (to adjacent characters) contribute to the balance.
                # The total number of such 'cost=1' moves can be sum of `s_counts` or `k`.
                # If `s_counts[i]` chars exist, and `k` is needed.
                # `abs(s_counts[i] - k)` needs to be resolved.
                # `min(abs(s_counts[i]-k), OFFSET)` is passed as balance.
                # The `OFFSET` needs to be greater than `k` or `max(s_counts[i])`.
                # This suggests `OFFSET` is `N`, but the crucial `min(abs(diff), 2)` means the `balance` is small.
                # Max accumulated flow of characters at cost 1 could be 26 (if all chars move to next).
                # To be safe, 2 * 26 = 52.
                
                OFFSET = 52
                # dp[j][balance_plus_offset] stores the min cost to transform,
                # considering characters up to current_char_code-1.
                # `j` is the number of target characters chosen so far.
                # `balance_plus_offset` is `balance + OFFSET`.
                
                dp = [[float('inf')] * (2 * OFFSET + 1) for _ in range(m + 1)]
                
                # Base case: before processing any character (index -1), 0 target characters chosen,
                # 0 balance, cost 0.
                dp[0][OFFSET] = 0 

                # Iterate through each character 'a' to 'z' (char_code 0 to 25)
                for char_code in range(26):
                    next_dp = [[float('inf')] * (2 * OFFSET + 1) for _ in range(m + 1)]
                    s_count = char_freq[char_code]

                    # Iterate over possible number of targets chosen so far (`j`)
                    for j in range(m + 1):
                        # Iterate over possible balance states from previous character
                        for b_offset_from_prev in range(2 * OFFSET + 1):
                            cost_so_far = dp[j][b_offset_from_prev]
                            if cost_so_far == float('inf'):
                                continue
                            
                            balance_from_prev = b_offset_from_prev - OFFSET
                            
                            # Effective count of current character, including carry-over from previous
                            # Positive balance_from_prev means excess characters from left, negative means deficit from left.
                            # `chars_to_use_from_left_for_cost_1` = amount of previous excess that can come to current char (cost 1).
                            # `chars_needed_for_deficit_from_left` = amount of previous deficit that needs current char (cost 1).
                            
                            chars_available_from_left = max(0, balance_from_prev)
                            chars_needed_for_left = max(0, -balance_from_prev)

                            # Cost of operations that are *forced* to be cost 2 (delete/insert)
                            # because they couldn't be resolved by cost 1 at previous steps.
                            # These are already incorporated into `cost_so_far`.
                            # We just need to manage `chars_available_from_left` and `chars_needed_for_left`.

                            # Option 1: Don't choose `char_code` as a target character (`j` remains the same)
                            # Target count for `char_code` is 0.
                            
                            # `temp_s_count`: total characters we have for current char_code + any excess from previous that passed through.
                            # If `balance_from_prev` is positive, we have `balance_from_prev` extra chars from `char_code-1`
                            # that can become `char_code` at cost 1.
                            # If `balance_from_prev` is negative, we need `abs(balance_from_prev)` chars for `char_code-1`.
                            
                            # `num_current_available = s_count + chars_available_from_left`
                            # `num_current_needed = chars_needed_for_left`
                            
                            # Combined pool of current s_count and leftover from previous that can be matched cost 1
                            effective_s_for_this_slot = s_count + balance_from_prev 
                            
                            # For Option 1 (Don't choose char_code as target):
                            # We want 0 chars of type `char_code`.
                            # Any `effective_s_for_this_slot` must be either passed to next or deleted.
                            
                            # If `effective_s_for_this_slot` is positive: excess chars.
                            # `excess_to_pass_right_or_cost2 = effective_s_for_this_slot`
                            ops_cost_2_option1 = 0
                            next_balance_option1 = 0
                            
                            if effective_s_for_this_slot > 0:
                                ops_cost_2_option1 = max(0, effective_s_for_this_slot - OFFSET) * 2
                                next_balance_option1 = min(effective_s_for_this_slot, OFFSET)
                            else: # deficit. must be filled by inserting new ones.
                                # This negative balance means we need to insert this many chars of this type,
                                # or obtain them from previous chars.
                                # Since we're not choosing this char, this deficit cannot be resolved directly here.
                                # It carries over to the right as deficit, which means `cost_so_far` has to absorb it.
                                ops_cost_2_option1 = max(0, -effective_s_for_this_slot - OFFSET) * 2 # Insert ops that are beyond OFFSET
                                next_balance_option1 = max(effective_s_for_this_slot, -OFFSET) # This is a negative balance passed.

                            # Update `next_dp` for Option 1
                            next_dp[j][next_balance_option1 + OFFSET] = min(
                                next_dp[j][next_balance_option1 + OFFSET],
                                cost_so_far + ops_cost_2_option1
                            )

                            # Option 2: Choose `char_code` as a target character (`j` becomes `j+1`)
                            if j < m: # Can only choose if we haven't reached `m` targets yet
                                # We need `k` occurrences of `char_code`.
                                `delta = effective_s_for_this_slot - k`
                                
                                ops_cost_2_option2 = 0
                                next_balance_option2 = 0

                                if delta > 0: # We have excess characters
                                    ops_cost_2_option2 = max(0, delta - OFFSET) * 2
                                    next_balance_option2 = min(delta, OFFSET)
                                else: # We have deficit characters
                                    ops_cost_2_option2 = max(0, -delta - OFFSET) * 2
                                    next_balance_option2 = max(delta, -OFFSET)
                                
                                # Update `next_dp` for Option 2
                                next_dp[j + 1][next_balance_option2 + OFFSET] = min(
                                    next_dp[j + 1][next_balance_option2 + OFFSET],
                                    cost_so_far + ops_cost_2_option2
                                )
                    dp = next_dp

                # After iterating through all characters 0-25 ('a' to 'z')
                # The final cost for this (m, k) pair is derived from `dp[m]`
                # Any remaining `balance` needs to be resolved by insertions or deletions (cost 1 per item).
                final_cost_for_mk = float('inf')
                for b_offset in range(2 * OFFSET + 1):
                    balance = b_offset - OFFSET
                    cost_for_remaining_balance = abs(balance) # Each unit costs 1 to insert/delete.
                    final_cost_for_mk = min(final_cost_for_mk, dp[m][b_offset] + cost_for_remaining_balance)
                
                # Add the base cost for length difference:
                # `abs(n - target_length)` is the cost of deleting/inserting characters that are not "matched".
                # The `final_cost_for_mk` covers the cost of changing matched characters.
                min_total_ops = min(min_total_ops, final_cost_for_mk + abs(n - target_length))
        
        return min_total_ops