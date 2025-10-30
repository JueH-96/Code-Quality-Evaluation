from typing import List

class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        """
        Finds the length of the longest valid substring of `word`.
        A substring is valid if none of its own substrings are in `forbidden`.
        """
        forbidden_set = set(forbidden)
        max_f_len = 0
        for s in forbidden:
            max_f_len = max(max_f_len, len(s))

        n = len(word)
        max_len = 0
        left_bound = 0

        for right in range(n):
            # We check for forbidden substrings ending at the current `right` pointer.
            # We only need to check substrings up to length `max_f_len`.
            # The start index 'j' of such substrings is in [right - max_f_len + 1, right].
            
            # For efficiency, we can iterate `j` from `right` downwards.
            # This checks suffixes of `word[0:right+1]` from length 1 up to `max_f_len`.
            start_check_idx = max(0, right - max_f_len + 1)
            
            for j in range(right, start_check_idx - 1, -1):
                substring = word[j : right + 1]
                if substring in forbidden_set:
                    # If `word[j:right+1]` is forbidden, any valid substring
                    # must start at an index greater than j.
                    # We update `left_bound` to reflect this new constraint.
                    left_bound = max(left_bound, j + 1)
                    
                    # Optimization: Since we iterate `j` from `right` downwards, the first
                    # forbidden substring we find imposes the strictest possible
                    # starting position requirement for this `right`. Any other forbidden
                    # suffix would start at an index `k < j`, giving a less strict
                    # bound `k+1 < j+1`. So, we can break after the first find.
                    break
            
            # The substring `word[left_bound:right+1]` is the longest valid
            # substring ending at the current `right`.
            current_len = right - left_bound + 1
            max_len = max(max_len, current_len)
            
        return max_len