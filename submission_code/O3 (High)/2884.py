from typing import List

class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        """
        Sliding–window solution.

        Because every forbidden string is no longer than 10 characters we only have
        to look at the last 10 characters that finish at the current position `r`.
        If one of those suffixes is forbidden, the window’s left border has to be
        moved so that the forbidden part is no longer inside the window.

        Complexity
        ----------
        n  = len(word)      (≤ 1e5)
        L  = max length of forbidden strings (= 10)

        For every position we examine at most L (=10) short substrings, giving
        time-complexity O(n · L)  ≈ 1 000 000 operations, and O(1) extra memory
        apart from the set that stores the forbidden words.
        """
        forbidden_set = set(forbidden)          # O(total length of forbidden)
        max_forb_len = 10                       # given by the constraints

        n = len(word)
        left = 0                                # left border of current valid window
        best = 0                                # best length found so far

        for right in range(n):                  # right border of the window
            # Check every suffix (length 1 … 10) that ends at position `right`
            for length in range(1, max_forb_len + 1):
                start = right - length + 1
                if start < left:      # already outside current window -> longer lengths will be as well
                    break
                if start < 0:         # ran off the string
                    break
                if word[start:right + 1] in forbidden_set:
                    # shift window so that this forbidden substring is no longer inside
                    left = start + 1
                    break             # no need to test longer lengths – they all include the same substring

            # update answer: current window is [left, right]
            best = max(best, right - left + 1)

        return best