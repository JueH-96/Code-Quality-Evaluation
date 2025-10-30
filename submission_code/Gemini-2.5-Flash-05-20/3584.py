from typing import List

class Solution:
    def validSequence(self, word1: str, word2: str) -> List[int]:
        n = len(word1)
        m = len(word2)

        # Handle edge case: if word2 is longer than word1, it's impossible to form.
        # (Though constraints state word2.length < word1.length)
        if m > n:
            return []
        
        # Precompute next_char: next_char[char_code][i] = smallest index j >= i where word1[j] == char_code
        # If no such index, it's n (representing out of bounds).
        # next_char[char_code][n] will always be n.
        next_char = [[n] * (n + 1) for _ in range(26)]
        for i in range(n - 1, -1, -1):
            for char_code in range(26):
                next_char[char_code][i] = next_char[char_code][i+1]
            next_char[ord(word1[i]) - ord('a')][i] = i

        # Precompute prev_char: prev_char[char_code][i] = largest index j <= i where word1[j] == char_code
        # If no such index, it's -1.
        prev_char = [[-1] * n for _ in range(26)]
        for i in range(n):
            if i > 0:
                for char_code in range(26):
                    prev_char[char_code][i] = prev_char[char_code][i-1]
            prev_char[ord(word1[i]) - ord('a')][i] = i

        # L_end_idx[k]: The index in word1 of the k-th char word2[k]
        # when forming the lexicographically smallest exact prefix word2[0..k].
        # Value is n if not possible to form up to this point.
        L_end_idx = [n] * m
        last_w1_idx = -1 # Start search from index 0 for word2[0]
        for k in range(m):
            next_w1_idx = next_char[ord(word2[k]) - ord('a')][last_w1_idx + 1]
            if next_w1_idx >= n:
                break # Cannot complete prefix from here onwards
            L_end_idx[k] = next_w1_idx
            last_w1_idx = next_w1_idx

        # R_furthest_match_idx[k]: The largest possible index `j` in word1 such that `word1[j]` is `word2[k]`
        # and it's possible to form the exact suffix `word2[k...m-1]` using characters from `word1[j:]`.
        # Value is -1 if not possible to form such a suffix starting from `k`.
        R_furthest_match_idx = [-1] * (m + 1)
        R_furthest_match_idx[m] = n # Sentinel: an empty suffix exists "after" index n-1.

        for k_rev in range(m - 1, -1, -1):
            target_char_code = ord(word2[k_rev]) - ord('a')
            
            # The character for word2[k_rev] must be at an index strictly less than R_furthest_match_idx[k_rev+1]
            # to leave room for the exact suffix match of word2[k_rev+1:]
            search_upper_bound = R_furthest_match_idx[k_rev+1] - 1
            
            if search_upper_bound < 0: # No space to search before the next suffix's start
                R_furthest_match_idx[k_rev] = -1
                continue
            
            match_idx = prev_char[target_char_code][search_upper_bound]
            
            # Ensure the found match_idx is valid (not -1) and allows enough characters before it
            # to accommodate the prefix (i.e. `k_rev` characters for `word2[0..k_rev-1]`)
            if match_idx == -1 or match_idx < k_rev:
                R_furthest_match_idx[k_rev] = -1
            else:
                R_furthest_match_idx[k_rev] = match_idx

        # Initialize best_result with the 0-difference case
        best_result = []
        is_possible_0_diff = True
        for k in range(m):
            if L_end_idx[k] == n: # If any part of the 0-diff prefix is impossible
                is_possible_0_diff = False
                break
            best_result.append(L_end_idx[k])
        
        if not is_possible_0_diff:
            best_result = None # Indicate no 0-diff solution

        # Iterate through each possible position `p` (0 to m-1) where the single difference occurs
        for p in range(m):
            # 1. Reconstruct prefix_indices for word2[0...p-1] (exact matches)
            prefix_indices = []
            last_prefix_char_w1_idx = -1
            can_form_prefix = True
            if p > 0:
                if L_end_idx[p-1] == n: # If the exact prefix up to p-1 was impossible
                    can_form_prefix = False
                else:
                    # Reconstruct the prefix path using next_char from L_end_idx values
                    curr_w1_idx = -1
                    for k_p in range(p):
                        curr_w1_idx = next_char[ord(word2[k_p]) - ord('a')][curr_w1_idx + 1]
                        prefix_indices.append(curr_w1_idx)
                    last_prefix_char_w1_idx = curr_w1_idx
            
            if not can_form_prefix:
                continue

            # 2. Find chosen_i_p for word2[p] with a difference
            # It must be strictly greater than last_prefix_char_w1_idx
            start_search_i_p = last_prefix_char_w1_idx + 1
            
            # It must be strictly less than R_furthest_match_idx[p+1]
            # to leave room for the exact suffix match.
            end_search_i_p = R_furthest_match_idx[p+1] - 1
            
            if end_search_i_p < start_search_i_p: # No valid range for chosen_i_p
                continue

            min_idx_for_diff = n # Sentinel for "no valid index found"
            for char_code in range(26):
                # Skip if current char_code matches word2[p] (we need a difference)
                if chr(ord('a') + char_code) == word2[p]:
                    continue
                
                # Find the next occurrence of this different character
                candidate_idx = next_char[char_code][start_search_i_p]
                
                # If this candidate index is within the valid search range
                if candidate_idx <= end_search_i_p:
                    min_idx_for_diff = min(min_idx_for_diff, candidate_idx)
            
            if min_idx_for_diff == n: # No valid `i_p` (differing character) found for this `p`
                continue
            
            chosen_i_p = min_idx_for_diff

            # 3. Reconstruct suffix_indices for word2[p+1...m-1] (exact matches)
            suffix_indices = []
            curr_w1_idx = chosen_i_p
            can_form_suffix = True
            for k_s in range(p + 1, m):
                next_suffix_match = next_char[ord(word2[k_s]) - ord('a')][curr_w1_idx + 1]
                if next_suffix_match >= n: # Cannot find next exact match for suffix
                    can_form_suffix = False
                    break
                suffix_indices.append(next_suffix_match)
                curr_w1_idx = next_suffix_match
            
            if not can_form_suffix:
                continue

            # Combine the parts to form the current candidate result
            current_candidate_result = prefix_indices + [chosen_i_p] + suffix_indices
            
            # Compare current_candidate_result with best_result found so far
            if best_result is None or current_candidate_result < best_result:
                best_result = current_candidate_result

        return best_result if best_result is not None else []