from typing import List

class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        if not words:
            return 0
        
        # Initialize with the first word's state
        first_char = words[0][0]
        last_char = words[0][-1]
        current_length = len(words[0])
        current_dp = {(first_char, last_char): current_length}
        
        for word in words[1:]:
            w_f = word[0]
            w_l = word[-1]
            w_len = len(word)
            next_dp = {}
            
            for (f, l), curr_len in current_dp.items():
                # Option 1: append the current string with the new word (current_str + word)
                new_f1, new_l1 = f, w_l
                overlap1 = 1 if l == w_f else 0
                new_len1 = curr_len + w_len - overlap1
                key1 = (new_f1, new_l1)
                if key1 in next_dp:
                    if new_len1 < next_dp[key1]:
                        next_dp[key1] = new_len1
                else:
                    next_dp[key1] = new_len1
                
                # Option 2: prepend the new word to current string (word + current_str)
                new_f2, new_l2 = w_f, l
                overlap2 = 1 if w_l == f else 0
                new_len2 = curr_len + w_len - overlap2
                key2 = (new_f2, new_l2)
                if key2 in next_dp:
                    if new_len2 < next_dp[key2]:
                        next_dp[key2] = new_len2
                else:
                    next_dp[key2] = new_len2
            
            current_dp = next_dp
        
        return min(current_dp.values()) if current_dp else 0