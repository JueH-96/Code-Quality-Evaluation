from typing import List

class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        # Preprocess forbidden strings by their lengths
        max_forbidden_len = 0
        forb_by_len = {}
        for s in forbidden:
            L = len(s)
            max_forbidden_len = max(max_forbidden_len, L)
            if L not in forb_by_len:
                forb_by_len[L] = set()
            forb_by_len[L].add(s)
        
        n = len(word)
        l = 0  # left boundary of the current valid window
        ans = 0
        
        # For each ending position i, try all forbidden lengths up to 10 (or i+1)
        for i in range(n):
            # Check substrings ending at i of length at most max_forbidden_len
            # If any is forbidden, move l just past the start of that forbidden substring
            for L in range(1, min(max_forbidden_len, i + 1) + 1):
                if L in forb_by_len:
                    if word[i - L + 1 : i + 1] in forb_by_len[L]:
                        # substring [i-L+1 .. i] is forbidden, so
                        # valid window must start at least at i-L+2
                        l = max(l, i - L + 2)
            # update answer with the length of the window [l .. i]
            ans = max(ans, i - l + 1)
        
        return ans