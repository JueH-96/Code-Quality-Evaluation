from typing import List

class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        if not words:
            return 0
        
        # Initialize the DP with the first word's start and end characters and its length
        first_word = words[0]
        dp = {(first_word[0], first_word[-1]): len(first_word)}
        
        for word in words[1:]:
            s_word = word[0]
            e_word = word[-1]
            len_word = len(word)
            next_dp = {}
            
            for (current_s, current_e), current_len in dp.items():
                # Option 1: Append the current word to the existing string
                new_s_append = current_s
                new_e_append = e_word
                overlap_append = 1 if current_e == s_word else 0
                total_len_append = current_len + len_word - overlap_append
                key_append = (new_s_append, new_e_append)
                if key_append in next_dp:
                    next_dp[key_append] = min(next_dp[key_append], total_len_append)
                else:
                    next_dp[key_append] = total_len_append
                
                # Option 2: Prepend the current word to the existing string
                new_s_prepend = s_word
                new_e_prepend = current_e
                overlap_prepend = 1 if e_word == current_s else 0
                total_len_prepend = current_len + len_word - overlap_prepend
                key_prepend = (new_s_prepend, new_e_prepend)
                if key_prepend in next_dp:
                    next_dp[key_prepend] = min(next_dp[key_prepend], total_len_prepend)
                else:
                    next_dp[key_prepend] = total_len_prepend
            
            dp = next_dp
        
        return min(dp.values())