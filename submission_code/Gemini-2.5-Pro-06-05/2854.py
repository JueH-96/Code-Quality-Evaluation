from typing import List

class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        """
        Calculates the minimum possible length of the final string after n-1 join operations.

        This problem is solved using dynamic programming. At each step `i`, we maintain a set of
        possible states, where each state is defined by the first and last characters of the
        concatenated string `str_i` and its minimum possible length.

        Let `dp` be a dictionary mapping a `(start_char_idx, end_char_idx)` tuple to the minimum
        length of a string formed so far with those start and end characters.
        """
        n = len(words)
        
        # dp: {(start_idx, end_idx): min_length}
        dp = {}
        
        # Base case: After processing the first word (i=0)
        w0 = words[0]
        f0_idx = ord(w0[0]) - ord('a')
        l0_idx = ord(w0[-1]) - ord('a')
        dp[(f0_idx, l0_idx)] = len(w0)

        # Iterate from the second word to the last
        for i in range(1, n):
            new_dp = {}
            wi = words[i]
            fi_idx = ord(wi[0]) - ord('a')
            li_idx = ord(wi[-1]) - ord('a')
            leni = len(wi)

            for (prev_f_idx, prev_l_idx), prev_len in dp.items():
                
                # Option 1: Append words[i] -> join(str_{i-1}, words[i])
                # The resulting string starts with the same char as str_{i-1}
                # and ends with the last char of words[i].
                len1 = prev_len + leni
                if prev_l_idx == fi_idx:
                    len1 -= 1
                
                key1 = (prev_f_idx, li_idx)
                if key1 not in new_dp or len1 < new_dp[key1]:
                    new_dp[key1] = len1

                # Option 2: Prepend words[i] -> join(words[i], str_{i-1})
                # The resulting string starts with the first char of words[i]
                # and ends with the same char as str_{i-1}.
                len2 = prev_len + leni
                if li_idx == prev_f_idx:
                    len2 -= 1
                
                key2 = (fi_idx, prev_l_idx)
                if key2 not in new_dp or len2 < new_dp[key2]:
                    new_dp[key2] = len2
            
            dp = new_dp
        
        # The final answer is the minimum of all possible lengths.
        return min(dp.values())