import math
from typing import List

class Solution:
  def minimizeConcatenatedLength(self, words: List[str]) -> int:
    n = len(words)
    
    infinity = math.inf
    # dp_prev[first_char_idx][last_char_idx] stores the minimum length
    # of the string formed by processing words up to the previous step (k-1),
    # such that the string starts with first_char_idx and ends with last_char_idx.
    # Characters 'a' through 'z' are mapped to indices 0 through 25.
    dp_prev = [[infinity] * 26 for _ in range(26)]

    # Base case: str_0 = words[0]
    # The string after processing words[0] is just words[0] itself.
    s0 = words[0]
    fc0 = ord(s0[0]) - ord('a') # First char of words[0]
    lc0 = ord(s0[-1]) - ord('a') # Last char of words[0]
    len0 = len(s0)              # Length of words[0]
    
    dp_prev[fc0][lc0] = len0
    
    # Iterate for k from 1 to n-1.
    # In iteration k, we are processing words[k] to form str_k using str_{k-1}.
    # dp_prev holds the results for str_{k-1}.
    # dp_curr will hold the results for str_k.
    for k in range(1, n):
        # Initialize dp_curr for the current step k.
        dp_curr = [[infinity] * 26 for _ in range(26)]
        current_word = words[k]
        curr_w_fc = ord(current_word[0]) - ord('a') # First char of words[k]
        curr_w_lc = ord(current_word[-1]) - ord('a') # Last char of words[k]
        curr_w_len = len(current_word)              # Length of words[k]

        for prev_fc_idx in range(26): # Iterating over possible first chars of str_{k-1}
            for prev_lc_idx in range(26): # Iterating over possible last chars of str_{k-1}
                
                # This is min_len(str_{k-1}) given its first char prev_fc_idx and last char prev_lc_idx
                len_str_prev = dp_prev[prev_fc_idx][prev_lc_idx] 
                
                if len_str_prev == infinity:
                    # This specific (first_char, last_char) combination for str_{k-1} was not reachable.
                    continue 

                # Option 1: str_k = join(str_{k-1}, current_word)
                # The first character of str_k will be prev_fc_idx (the first char of str_{k-1}).
                # The last character of str_k will be curr_w_lc (the last char of current_word).
                new_len_opt1 = len_str_prev + curr_w_len
                if prev_lc_idx == curr_w_fc: # Last char of str_{k-1} matches first char of current_word
                    new_len_opt1 -= 1
                
                # Update dp_curr for the string formed by Option 1.
                # This string starts with prev_fc_idx and ends with curr_w_lc.
                dp_curr[prev_fc_idx][curr_w_lc] = min(dp_curr[prev_fc_idx][curr_w_lc], new_len_opt1)

                # Option 2: str_k = join(current_word, str_{k-1})
                # The first character of str_k will be curr_w_fc (the first char of current_word).
                # The last character of str_k will be prev_lc_idx (the last char of str_{k-1}).
                new_len_opt2 = curr_w_len + len_str_prev
                if curr_w_lc == prev_fc_idx: # Last char of current_word matches first char of str_{k-1}
                    new_len_opt2 -= 1
                
                # Update dp_curr for the string formed by Option 2.
                # This string starts with curr_w_fc and ends with prev_lc_idx.
                dp_curr[curr_w_fc][prev_lc_idx] = min(dp_curr[curr_w_fc][prev_lc_idx], new_len_opt2)
        
        # The results from this iteration (dp_curr, for str_k) become the
        # 'previous' results (dp_prev) for the next iteration (to form str_{k+1}).
        dp_prev = dp_curr 

    # After all words are processed, dp_prev holds the results for str_{n-1}.
    # The final answer is the minimum length found in any state of dp_prev.
    min_final_len = infinity
    for r_idx in range(26):
        for c_idx in range(26):
            min_final_len = min(min_final_len, dp_prev[r_idx][c_idx])
    
    # Given n >= 1, at least one path exists, so min_final_len will be a finite number.
    return min_final_len