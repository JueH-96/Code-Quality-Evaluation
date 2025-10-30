class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        if not words:
            return 0
        
        # Initialize the DP with the first word's state
        first_word = words[0]
        f = first_word[0]
        l = first_word[-1]
        prev_dp = {(f, l): len(first_word)}
        
        for word in words[1:]:
            current_dp = {}
            for (f_prev, l_prev), len_prev in prev_dp.items():
                # Prepend the current word
                new_f = word[0]
                new_l = l_prev
                overlap = 1 if word[-1] == f_prev else 0
                new_len = len_prev + len(word) - overlap
                if (new_f, new_l) in current_dp:
                    if new_len < current_dp[(new_f, new_l)]:
                        current_dp[(new_f, new_l)] = new_len
                else:
                    current_dp[(new_f, new_l)] = new_len
                
                # Append the current word
                new_f_append = f_prev
                new_l_append = word[-1]
                overlap_append = 1 if l_prev == word[0] else 0
                new_len_append = len_prev + len(word) - overlap_append
                if (new_f_append, new_l_append) in current_dp:
                    if new_len_append < current_dp[(new_f_append, new_l_append)]:
                        current_dp[(new_f_append, new_l_append)] = new_len_append
                else:
                    current_dp[(new_f_append, new_l_append)] = new_len_append
            
            prev_dp = current_dp.copy()
        
        return min(prev_dp.values()) if prev_dp else 0