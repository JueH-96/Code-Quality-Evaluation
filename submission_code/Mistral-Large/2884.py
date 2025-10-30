from typing import List

class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        # Create a set for quick lookup of forbidden substrings
        forbidden_set = set(forbidden)

        # Initialize variables to track the maximum length of valid substring
        max_length = 0
        left = 0

        # Iterate over the word with the right pointer
        for right in range(len(word)):
            # Check all substrings ending at the current `right` index
            for k in range(right - left + 1):
                if word[right - k:right + 1] in forbidden_set:
                    # Move the left pointer to the right of the forbidden substring
                    left = right - k + 1
                    break
            # Calculate the current valid substring length
            max_length = max(max_length, right - left + 1)

        return max_length