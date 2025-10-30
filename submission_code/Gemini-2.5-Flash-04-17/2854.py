import sys
from typing import List

class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        n = len(words)
        # Constraints: 1 <= words.length <= 1000

        # dp[f][l] stores the minimum length of a string formed by
        # concatenating words[0]...words[i] that starts with character
        # represented by index f and ends with character represented by index l.
        # f and l are 0-25 for 'a'-'z'.
        
        # Initialize dp table for the first word (i = 0)
        # We use a 26x26 table. Initialize with a large value representing infinity.
        dp = [[sys.maxsize] * 26 for _ in range(26)]

        first_char_idx = ord(words[0][0]) - ord('a')
        last_char_idx = ord(words[0][-1]) - ord('a')
        dp[first_char_idx][last_char_idx] = len(words[0])

        # Iterate through the remaining words from i = 1 to n-1
        for i in range(1, n):
            next_word = words[i]
            next_word_len = len(next_word)
            next_word_first_char_idx = ord(next_word[0]) - ord('a')
            next_word_last_char_idx = ord(next_word[-1]) - ord('a')

            # Create a new DP table for the current step (words[0]...words[i])
            # Initialize with a large value representing infinity.
            new_dp = [[sys.maxsize] * 26 for _ in range(26)]

            # Iterate through all possible previous states (first_char, last_char) from dp (words[0]...words[i-1])
            for prev_first_idx in range(26):
                for prev_last_idx in range(26):
                    # Only consider reachable previous states (those not initialized to infinity)
                    if dp[prev_first_idx][prev_last_idx] != sys.maxsize:
                        prev_len = dp[prev_first_idx][prev_last_idx]

                        # Option 1: join(prev_str, next_word)
                        # The previous string ends with char corresponding to prev_last_idx.
                        # The next word starts with char corresponding to next_word_first_char_idx.
                        # Overlap occurs if prev_last_idx == next_word_first_char_idx.
                        overlap1 = 1 if prev_last_idx == next_word_first_char_idx else 0
                        new_len1 = prev_len + next_word_len - overlap1
                        
                        # The new string starts with char corresponding to prev_first_idx
                        # and ends with char corresponding to next_word_last_char_idx.
                        new_first_idx1 = prev_first_idx
                        new_last_idx1 = next_word_last_char_idx
                        # Update the new_dp table with the minimum length found so far for this state
                        new_dp[new_first_idx1][new_last_idx1] = min(new_dp[new_first_idx1][new_last_idx1], new_len1)

                        # Option 2: join(next_word, prev_str)
                        # The next word ends with char corresponding to next_word_last_char_idx.
                        # The previous string starts with char corresponding to prev_first_idx.
                        # Overlap occurs if next_word_last_char_idx == prev_first_idx.
                        overlap2 = 1 if next_word_last_char_idx == prev_first_idx else 0
                        new_len2 = next_word_len + prev_len - overlap2

                        # The new string starts with char corresponding to next_word_first_char_idx
                        # and ends with char corresponding to prev_last_idx.
                        new_first_idx2 = next_word_first_char_idx
                        new_last_idx2 = prev_last_idx
                        # Update the new_dp table with the minimum length found so far for this state
                        new_dp[new_first_idx2][new_last_idx2] = min(new_dp[new_first_idx2][new_last_idx2], new_len2)
            
            # The new_dp table is now complete for words[0]...words[i].
            # Replace the old dp table with the new one for the next iteration (words[0]...words[i+1]).
            dp = new_dp

        # After processing all words (up to words[n-1]), the minimum length is the minimum value
        # across all possible first and last characters in the final dp table.
        min_length = sys.maxsize
        for first_idx in range(26):
            for last_idx in range(26):
                min_length = min(min_length, dp[first_idx][last_idx])

        return min_length