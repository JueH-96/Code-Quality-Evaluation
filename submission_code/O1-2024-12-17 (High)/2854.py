class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        import math
        
        # We'll use a dynamic programming (DP) approach where the state encodes:
        #   1) how many words have been used so far (i),
        #   2) the first character of the resulting joined string,
        #   3) the last character of the resulting joined string.
        #
        # However, storing a 3D DP table of shape [n][26][26] could be large
        # (up to 1000 * 26 * 26), but still feasible in Python if implemented carefully.
        #
        # But to save memory, we only need DP for the current step and the next step
        # because we transition DP[i] -> DP[i+1]. This way, each DP layer is just 26*26 = 676.
        #
        # DP definition:
        #   dp[fc*26 + lc] = minimal length of the string formed after incorporating
        #                    i words, having first character = fc, last character = lc
        # where fc, lc are in [0..25].
        #
        # Transition (when adding words[i]):
        #
        #   Let L = dp[current_state] (some int),
        #       w = words[i],
        #       wfc = first char of w, wlc = last char of w,
        #       lenw = len(w).
        #
        #   We have two ways to join:
        #     1) str_{i-1} + w
        #        new_length = L + lenw - (1 if (last_char_of_str_{i-1} == first_char_of_w) else 0)
        #        first_char_of_new_string = (old first_char)
        #        last_char_of_new_string = wlc
        #
        #     2) w + str_{i-1}
        #        new_length = L + lenw - (1 if (last_char_of_w == first_char_of_str_{i-1}) else 0)
        #        first_char_of_new_string = wfc
        #        last_char_of_new_string = (old last_char)
        #
        # We'll compute these transitions for each possible state (fc, lc) that is not infinity.
        # At the end, dp after using all n words gives us all possible (fc, lc) pairs. The answer
        # is the minimum of dp over all (fc, lc).
        
        # Helper to convert a character to an index [0..25].
        def char_to_index(c: str) -> int:
            return ord(c) - ord('a')
        
        # Fast indexing for (fc, lc) into dp array of length 26*26.
        def idx(fc: int, lc: int) -> int:
            return fc * 26 + lc
        
        n = len(words)
        if n == 1:
            # If there's only one word, the answer is just its length.
            return len(words[0])
        
        # Initialize DP array of size 26*26 for "after using the first word".
        INF = 10**9
        dp = [INF] * (26 * 26)
        
        # First word
        first_word = words[0]
        fc0 = char_to_index(first_word[0])
        lc0 = char_to_index(first_word[-1])
        dp[idx(fc0, lc0)] = len(first_word)  # length of the initial string
        
        # Iterate over the remaining words, one by one
        for i in range(1, n):
            w = words[i]
            wfc = char_to_index(w[0])
            wlc = char_to_index(w[-1])
            lenw = len(w)
            
            next_dp = [INF] * (26 * 26)
            
            for pair in range(26 * 26):
                cur_len = dp[pair]
                if cur_len == INF:
                    continue
                
                # Decode the old first char and last char
                old_fc = pair // 26
                old_lc = pair % 26
                
                # 1) str_{i-1} + w
                new_len = cur_len + lenw
                if old_lc == wfc:  # overlap of 1 char
                    new_len -= 1
                # new string's first char stays old_fc; last char becomes wlc
                new_fc = old_fc
                new_lc = wlc
                idx_new = idx(new_fc, new_lc)
                if new_len < next_dp[idx_new]:
                    next_dp[idx_new] = new_len
                
                # 2) w + str_{i-1}
                new_len = cur_len + lenw
                if wlc == old_fc:  # overlap of 1 char
                    new_len -= 1
                # new string's first char becomes wfc; last char stays old_lc
                new_fc = wfc
                new_lc = old_lc
                idx_new = idx(new_fc, new_lc)
                if new_len < next_dp[idx_new]:
                    next_dp[idx_new] = new_len
            
            dp = next_dp
        
        # The answer is the minimum value in dp after using all words.
        return min(dp)