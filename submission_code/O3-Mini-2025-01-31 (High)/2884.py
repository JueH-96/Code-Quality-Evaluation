from typing import List

class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        # Convert forbidden list to a set for O(1) membership tests.
        forb_set = set(forbidden)
        n = len(word)
        ans = 0
        i = 0  # Left pointer for our sliding window
        for j in range(n):
            # For each new character added at index j, check if any
            # forbidden substring ends at j by examining at most 10 recent characters.
            for k in range(1, 11):  # k is the length of the candidate substring ending at j.
                start = j - k + 1
                if start < i:  # The substring would not be fully within the current window.
                    break
                # If the substring from start to j is forbidden, move the left pointer.
                if word[start: j + 1] in forb_set:
                    i = max(i, start + 1)
            # The current window [i, j] is now valid.
            ans = max(ans, j - i + 1)
        return ans