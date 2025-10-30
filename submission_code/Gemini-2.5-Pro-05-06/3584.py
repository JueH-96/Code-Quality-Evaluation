import sys

class Solution:
    def validSequence(self, word1: str, word2: str) -> list[int]:
        n = len(word1)
        m = len(word2)

        # Sentinel values:
        # n: for indices >= k, means "not found" or "past end of word1"
        # -1: for indices < k, means "not found"

        # Precomputation
        next_char_idx = [[n] * (n + 1) for _ in range(26)]
        for char_ord in range(26):
            for k_idx in range(n - 1, -1, -1):
                next_char_idx[char_ord][k_idx] = next_char_idx[char_ord][k_idx+1]
                if ord(word1[k_idx]) - ord('a') == char_ord:
                    next_char_idx[char_ord][k_idx] = k_idx
        
        prev_char_idx = [[-1] * (n + 1) for _ in range(26)]
        for char_ord in range(26):
            for k_idx in range(1, n + 1): # k_idx refers to "up to (but not including) index k_idx"
                prev_char_idx[char_ord][k_idx] = prev_char_idx[char_ord][k_idx-1]
                if ord(word1[k_idx-1]) - ord('a') == char_ord: # word1[k_idx-1] is char at index k_idx-1
                    prev_char_idx[char_ord][k_idx] = k_idx-1

        next_any_char_except = [[n] * (n + 1) for _ in range(26)]
        for char_ord_target in range(26):
            for k_idx in range(n - 1, -1, -1):
                next_any_char_except[char_ord_target][k_idx] = next_any_char_except[char_ord_target][k_idx+1]
                if ord(word1[k_idx]) - ord('a') != char_ord_target:
                    next_any_char_except[char_ord_target][k_idx] = k_idx

        prev_any_char_except = [[-1] * (n + 1) for _ in range(26)]
        for char_ord_target in range(26):
            for k_idx in range(1, n + 1):
                prev_any_char_except[char_ord_target][k_idx] = prev_any_char_except[char_ord_target][k_idx-1]
                if ord(word1[k_idx-1]) - ord('a') != char_ord_target:
                    prev_any_char_except[char_ord_target][k_idx] = k_idx-1
        
        min_first_idx_for_suffix = [[n] * 2 for _ in range(m + 1)]
        min_first_idx_for_suffix[m][0] = n 
        min_first_idx_for_suffix[m][1] = n 

        for i in range(m - 1, -1, -1):
            target_char_ord = ord(word2[i]) - ord('a')

            # Calculate min_first_idx_for_suffix[i][0]
            limit_idx0 = min_first_idx_for_suffix[i+1][0]
            # To form suffix word2[i...m-1] with 0 mismatches:
            # word1[p] must match word2[i], and p < (index for word2[i+1] with 0 mismatches)
            p0 = prev_char_idx[target_char_ord][limit_idx0]
            if p0 != -1:
                min_first_idx_for_suffix[i][0] = p0
            
            # Calculate min_first_idx_for_suffix[i][1]
            # Case A: word1[p] matches word2[i]. 1 mismatch in word2[i+1...m-1].
            p_candA = n 
            limit_idx1_A = min_first_idx_for_suffix[i+1][1]
            p_A = prev_char_idx[target_char_ord][limit_idx1_A]
            if p_A != -1:
                p_candA = p_A
            
            # Case B: word1[p] mismatches word2[i]. 0 mismatches in word2[i+1...m-1].
            p_candB = n
            limit_idx0_B = min_first_idx_for_suffix[i+1][0]
            p_B = prev_any_char_except[target_char_ord][limit_idx0_B]
            if p_B != -1:
                p_candB = p_B
            
            min_first_idx_for_suffix[i][1] = min(p_candA, p_candB)

        ans = []
        last_selected_w1_idx = -1
        mismatches_made = 0

        for k in range(m): 
            current_target_char_ord = ord(word2[k]) - ord('a')
            min_candidate_idx_w1 = last_selected_w1_idx + 1
            max_candidate_idx_w1 = n - (m - k)
            
            chosen_idx_k = n 

            # Option 1: word1[idx_k] == word2[k]
            cand_p_match = next_char_idx[current_target_char_ord][min_candidate_idx_w1]
            if cand_p_match < n and cand_p_match <= max_candidate_idx_w1:
                valid_suffix_possible = False
                if k + 1 == m: # Suffix is empty
                    valid_suffix_possible = True
                else: # Suffix is non-empty
                    idx_for_k_plus_1 = min_first_idx_for_suffix[k+1][mismatches_made]
                    if idx_for_k_plus_1 < n and idx_for_k_plus_1 > cand_p_match:
                        valid_suffix_possible = True
                
                if valid_suffix_possible:
                    chosen_idx_k = min(chosen_idx_k, cand_p_match)

            # Option 2: word1[idx_k] != word2[k]
            if mismatches_made == 0:
                cand_p_mismatch = next_any_char_except[current_target_char_ord][min_candidate_idx_w1]
                if cand_p_mismatch < n and cand_p_mismatch <= max_candidate_idx_w1:
                    valid_suffix_possible = False
                    # Mismatch here, so suffix word2[k+1...m-1] needs 0 mismatches
                    if k + 1 == m: # Suffix is empty
                        valid_suffix_possible = True
                    else: # Suffix is non-empty
                        idx_for_k_plus_1 = min_first_idx_for_suffix[k+1][0]
                        if idx_for_k_plus_1 < n and idx_for_k_plus_1 > cand_p_mismatch:
                            valid_suffix_possible = True
                    
                    if valid_suffix_possible:
                        chosen_idx_k = min(chosen_idx_k, cand_p_mismatch)
            
            if chosen_idx_k == n: 
                return []
            
            ans.append(chosen_idx_k)
            
            if ord(word1[chosen_idx_k]) - ord('a') != current_target_char_ord:
                mismatches_made = 1
            
            last_selected_w1_idx = chosen_idx_k
            
        return ans