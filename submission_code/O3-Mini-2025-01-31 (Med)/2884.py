from typing import List

class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        # Convert forbidden into a set for O(1) lookups.
        forbidden_set = set(forbidden)
        n = len(word)
        left = 0
        ans = 0

        # For each end index of the current window.
        for i in range(n):
            # We'll compute the furthest position we need to slide the left pointer
            # If any substring ending at index i of length L (up to 10) is forbidden,
            # then that substring starts at index (i - L + 1).
            # To remove that forbidden pattern from the window, we need to set:
            #    left > (i - L + 1), i.e. left = i - L + 2 at least.
            new_left = left  # candidate new left pointer
            # Only need to check at most 10 characters ending at i
            for L in range(1, min(10, i + 1) + 1):
                # substring ending at i and of length L
                substr = word[i - L + 1 : i + 1]
                if substr in forbidden_set:
                    # We shift left pointer to one position after the start of this forbidden substring.
                    candidate_left = i - L + 2
                    if candidate_left > new_left:
                        new_left = candidate_left
            left = new_left
            # Update answer with current valid window length.
            ans = max(ans, i - left + 1)
        return ans