import math
from typing import List

class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        INF = 10**9
        n = len(words)
        if n == 0:
            return 0  # Though constraints ensure n >= 1, handle edge case
        
        # Precompute word data: length, first char index, last char index
        word_data = []
        for word in words:
            len_w = len(word)
            first_idx = ord(word[0]) - ord('a')
            last_idx = ord(word[-1]) - ord('a')
            word_data.append((len_w, first_idx, last_idx))
        
        # Initialize dp for str_0 (after adding words[0])
        dp_curr = [[INF for _ in range(26)] for _ in range(26)]
        len_w0, first_c, last_c = word_data[0]
        dp_curr[first_c][last_c] = len_w0
        
        # For each subsequent word from index 1 to n-1
        for i in range(1, n):
            dp_next = [[INF for _ in range(26)] for _ in range(26)]
            len_word, word_first_idx, word_last_idx = word_data[i]
            
            # Iterate over all possible previous first and last char indices
            for prev_first in range(26):
                for prev_last in range(26):
                    if dp_curr[prev_first][prev_last] < INF:  # Valid previous state
                        # Option 1: Append words[i] to current string
                        save_append = 1 if prev_last == word_first_idx else 0
                        new_len_append = dp_curr[prev_first][prev_last] + len_word - save_append
                        new_first_append = prev_first
                        new_last_append = word_last_idx
                        if dp_next[new_first_append][new_last_append] > new_len_append:
                            dp_next[new_first_append][new_last_append] = new_len_append
                        
                        # Option 2: Prepend words[i] to current string
                        save_prepend = 1 if word_last_idx == prev_first else 0
                        new_len_prepend = dp_curr[prev_first][prev_last] + len_word - save_prepend
                        new_first_prepend = word_first_idx
                        new_last_prepend = prev_last
                        if dp_next[new_first_prepend][new_last_prepend] > new_len_prepend:
                            dp_next[new_first_prepend][new_last_prepend] = new_len_prepend
            
            # Update dp_curr to dp_next for the next iteration
            dp_curr = dp_next
        
        # The minimum length is the minimum value in dp_curr
        min_length = INF
        for first in range(26):
            for last in range(26):
                if min_length > dp_curr[first][last]:
                    min_length = dp_curr[first][last]
        return min_length