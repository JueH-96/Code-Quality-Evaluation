class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        if not words:
            return 0
        
        # Initialize with the first word's first and last characters and its length
        first_char = words[0][0]
        last_char = words[0][-1]
        dp = {(first_char, last_char): len(words[0])}
        
        for word in words[1:]:
            next_dp = {}
            word_first = word[0]
            word_last = word[-1]
            word_len = len(word)
            
            for (current_first, current_last), current_total in dp.items():
                # Option 1: Prepend the current string with the word
                # i.e., new_str = word + current_str
                overlap_prepend = 1 if word_last == current_first else 0
                new_len_prepend = current_total + word_len - overlap_prepend
                new_first_prepend = word_first
                new_last_prepend = current_last
                key_prepend = (new_first_prepend, new_last_prepend)
                if key_prepend in next_dp:
                    if new_len_prepend < next_dp[key_prepend]:
                        next_dp[key_prepend] = new_len_prepend
                else:
                    next_dp[key_prepend] = new_len_prepend
                
                # Option 2: Append the word to the current string
                # i.e., new_str = current_str + word
                overlap_append = 1 if current_last == word_first else 0
                new_len_append = current_total + word_len - overlap_append
                new_first_append = current_first
                new_last_append = word_last
                key_append = (new_first_append, new_last_append)
                if key_append in next_dp:
                    if new_len_append < next_dp[key_append]:
                        next_dp[key_append] = new_len_append
                else:
                    next_dp[key_append] = new_len_append
            
            dp = next_dp
        
        return min(dp.values()) if dp else 0