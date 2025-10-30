from typing import List

class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        if not words:
            return 0
        # Initialize DP with the first word
        first_word = words[0]
        f = first_word[0]
        l = first_word[-1]
        dp = { (f, l): len(first_word) }
        
        for word in words[1:]:
            new_dp = {}
            w_f = word[0]
            w_l = word[-1]
            w_len = len(word)
            for (prev_f, prev_l), prev_len in dp.items():
                # Append the current word to previous string
                new_f_app = prev_f
                new_l_app = w_l
                overlap_app = 1 if prev_l == w_f else 0
                new_len_app = prev_len + w_len - overlap_app
                key_app = (new_f_app, new_l_app)
                if key_app in new_dp:
                    if new_len_app < new_dp[key_app]:
                        new_dp[key_app] = new_len_app
                else:
                    new_dp[key_app] = new_len_app
                # Prepend the current word to previous string
                new_f_pre = w_f
                new_l_pre = prev_l
                overlap_pre = 1 if w_l == prev_f else 0
                new_len_pre = prev_len + w_len - overlap_pre
                key_pre = (new_f_pre, new_l_pre)
                if key_pre in new_dp:
                    if new_len_pre < new_dp[key_pre]:
                        new_dp[key_pre] = new_len_pre
                else:
                    new_dp[key_pre] = new_len_pre
            dp = new_dp
        return min(dp.values())