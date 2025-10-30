import math

class Solution:
    # Helper method to check if a substring represented by its character counts is balanced.
    # A string is balanced if every character that appears in the string occurs the same number of times.
    def _is_balanced(self, counts: list[int]) -> bool:
        target_freq = 0 # Stores the frequency of the first character encountered.
                        # Initialized to 0, meaning no character with positive count found yet.
        
        for count_val in counts: # Iterate through the counts of 'a' through 'z'.
            if count_val > 0: # If this character is present in the substring
                if target_freq == 0: # This is the first character type found
                    target_freq = count_val
                elif count_val != target_freq: # This character has a different frequency
                    return False # Not balanced
        
        # If the loop completes, one of two conditions holds:
        # 1. All characters present in the substring have the same frequency (target_freq).
        # 2. The substring was empty (no counts > 0, target_freq remains 0).
        #    (Our main logic ensures we only call this for non-empty substrings,
        #     so target_freq will be > 0 if called from the main loop.)
        return True

    def minimumSubstringsInPartition(self, s: str) -> int:
        n = len(s)
        
        # dp[k] stores the minimum number of balanced substrings for the prefix s[0...k-1].
        # The length of this prefix is k.
        dp = [0] * (n + 1) 
        # dp[0] = 0: an empty prefix requires 0 substrings. This is the base case.

        # Iterate `i` from 1 to `n`. `i` represents the length of the prefix s[0...i-1].
        for i in range(1, n + 1):
            # Initialize dp[i] to infinity. math.inf is a float; the result needs to be int.
            # The maximum possible partitions is `n` (if each char is its own substring).
            dp[i] = math.inf 
            
            # `current_substring_counts` stores character frequencies for s[j...i-1].
            current_substring_counts = [0] * 26 
            
            # Iterate `j` downwards from `i-1` to `0`.
            # `j` is the 0-indexed start of the potential last substring s[j...i-1].
            # (In Python slice notation, this substring is s[j:i]).
            for j in range(i - 1, -1, -1):
                # Add character s[j] to the current substring being considered.
                # (The substring effectively grows by prepending s[j] as j decreases).
                char_idx = ord(s[j]) - ord('a')
                current_substring_counts[char_idx] += 1
                
                # Check if the current substring s[j...i-1] (represented by current_substring_counts) is balanced.
                if self._is_balanced(current_substring_counts):
                    # If s[j...i-1] is balanced, it can be the last substring in a partition of s[0...i-1].
                    # The prefix s[0...j-1] must have been partitioned in dp[j] ways.
                    # So, total substrings for s[0...i-1] using this choice is dp[j] + 1.
                    # We want the minimum such value.
                    # dp[j] is guaranteed to be finite because dp[0]=0, and any prefix can be partitioned
                    # (e.g., into single-character substrings, which are always balanced).
                    dp[i] = min(dp[i], dp[j] + 1)
        
        # dp[n] contains the minimum number of balanced substrings for the entire string s.
        # This value will be finite (between 1 and n, inclusive for n>=1).
        return int(dp[n])