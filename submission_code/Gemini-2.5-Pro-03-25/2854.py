import math # math module is not strictly needed since float('inf') is built-in, but good practice maybe? Actually removed it.
from typing import List

class Solution:
  def minimizeConcatenatedLength(self, words: List[str]) -> int:
    """
    Calculates the minimum possible length of the final string formed by joining
    the words in the list `words`. At each step `i`, `words[i]` can be joined 
    either to the left or right of the string formed so far using `words[0...i-1]`.
    The join operation merges the strings, removing one character if the boundary
    characters match. Uses dynamic programming with space optimization.

    Args:
        words: A list of strings.
    Returns:
        An integer denoting the minimum possible length of the final concatenated string.
    """
    n = len(words)
    
    # Base case: If there's only one word, the result is its length.
    # Problem constraints state n >= 1, so n=0 case is not possible.
    if n == 1:
        return len(words[0])

    # Initialize DP table for the previous step (i-1).
    # dp_prev[p][q] stores the minimum length of the concatenated string formed using
    # words[0...i-1], such that the resulting string starts with character corresponding 
    # to index p and ends with character corresponding to index q.
    # Indices p, q range from 0 to 25, mapping 'a' to 0, 'b' to 1, ..., 'z' to 25.
    # Initialize all lengths to infinity.
    dp_prev = [[float('inf')] * 26 for _ in range(26)]

    # Base case: Process the first word (index 0)
    word0 = words[0]
    len0 = len(word0)
    first0_char = word0[0]
    last0_char = word0[-1]
    # Map characters to 0-25 indices
    first0_idx = ord(first0_char) - ord('a')
    last0_idx = ord(last0_char) - ord('a')
    
    # Set the initial state after processing the first word.
    dp_prev[first0_idx][last0_idx] = len0

    # DP transitions: Iterate through the words from the second word (index 1) up to the last word (index n-1).
    for i in range(1, n):
        # Initialize DP table for the current step (i). This will store minimum lengths after processing words[0...i].
        dp_curr = [[float('inf')] * 26 for _ in range(26)]
        current_word = words[i]
        len_curr = len(current_word)
        s_char = current_word[0]   # first character of current_word
        t_char = current_word[-1]  # last character of current_word
        s_idx = ord(s_char) - ord('a') # index for first character
        t_idx = ord(t_char) - ord('a') # index for last character

        # Iterate through all possible states from the previous step (i-1).
        # A state is defined by the pair of indices (p_idx, q_idx) for the first and last characters.
        for p_idx in range(26):
            for q_idx in range(26):
                # Check if the state (p_idx, q_idx) was reachable in the previous step (i.e., has finite length).
                if dp_prev[p_idx][q_idx] != float('inf'):
                    # L is the minimum length found for a string formed from words[0...i-1]
                    # starting with character p_idx and ending with character q_idx.
                    L = dp_prev[p_idx][q_idx] 
                    
                    # Retrieve the actual characters for boundary comparison check during join operation.
                    # Needed because the join rule depends on character equality.
                    q_char = chr(q_idx + ord('a')) # Last character of the string from previous step
                    p_char = chr(p_idx + ord('a')) # First character of the string from previous step

                    # Option 1: Perform join(previous_string, current_word)
                    # The previous string has length L, starts with p_char, ends with q_char.
                    # The current word has length len_curr, starts with s_char, ends with t_char.
                    # The boundary for join is between q_char (end of previous) and s_char (start of current).
                    # If q_char == s_char, length reduces by 1.
                    cost1 = 1 if q_char == s_char else 0
                    new_len1 = L + len_curr - cost1
                    # The resulting string starts with the same first character as previous_string (p_char).
                    new_first1_idx = p_idx
                    # The resulting string ends with the same last character as current_word (t_char).
                    new_last1_idx = t_idx
                    # Update the DP table for the current step with the minimum length found for this new state.
                    dp_curr[new_first1_idx][new_last1_idx] = min(dp_curr[new_first1_idx][new_last1_idx], new_len1)

                    # Option 2: Perform join(current_word, previous_string)
                    # The current word has length len_curr, starts with s_char, ends with t_char.
                    # The previous string has length L, starts with p_char, ends with q_char.
                    # The boundary for join is between t_char (end of current) and p_char (start of previous).
                    # If t_char == p_char, length reduces by 1.
                    cost2 = 1 if t_char == p_char else 0
                    new_len2 = len_curr + L - cost2
                    # The resulting string starts with the same first character as current_word (s_char).
                    new_first2_idx = s_idx
                    # The resulting string ends with the same last character as previous_string (q_char).
                    new_last2_idx = q_idx
                    # Update the DP table for the current step with the minimum length found for this new state.
                    dp_curr[new_first2_idx][new_last2_idx] = min(dp_curr[new_first2_idx][new_last2_idx], new_len2)
        
        # After computing all possible states for step i, update dp_prev to dp_curr for the next iteration.
        # This implements the space optimization, using only two layers of DP states.
        dp_prev = dp_curr 

    # After the loop finishes processing all words up to n-1, dp_prev holds the minimum lengths
    # for all possible final states (defined by first and last characters).
    # Find the overall minimum length among all possible final states.
    min_len = float('inf')
    for p_idx in range(26):
        for q_idx in range(26):
            min_len = min(min_len, dp_prev[p_idx][q_idx])
    
    # Return the overall minimum length found.
    # Since n>=1 is guaranteed by constraints, min_len will be a finite number.
    return min_len