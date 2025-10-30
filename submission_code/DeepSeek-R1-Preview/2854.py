class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        if not words:
            return 0
        
        # Initialize DP with the first word
        first_word = words[0]
        dp = {(first_word[0], first_word[-1]): len(first_word)}
        
        for i in range(1, len(words)):
            current_word = words[i]
            w_first = current_word[0]
            w_last = current_word[-1]
            w_len = len(current_word)
            new_dp = {}
            
            for (prev_first, prev_last), prev_total in dp.items():
                # Option 1: Add current_word to the end
                if prev_last == w_first:
                    new_len1 = prev_total + w_len - 1
                else:
                    new_len1 = prev_total + w_len
                new_first1, new_last1 = prev_first, w_last
                if (new_first1, new_last1) in new_dp:
                    if new_len1 < new_dp[(new_first1, new_last1)]:
                        new_dp[(new_first1, new_last1)] = new_len1
                else:
                    new_dp[(new_first1, new_last1)] = new_len1
                
                # Option 2: Add current_word to the front
                if w_last == prev_first:
                    new_len2 = prev_total + w_len - 1
                else:
                    new_len2 = prev_total + w_len
                new_first2, new_last2 = w_first, prev_last
                if (new_first2, new_last2) in new_dp:
                    if new_len2 < new_dp[(new_first2, new_last2)]:
                        new_dp[(new_first2, new_last2)] = new_len2
                else:
                    new_dp[(new_first2, new_last2)] = new_len2
            
            dp = new_dp
        
        return min(dp.values()) if dp else 0