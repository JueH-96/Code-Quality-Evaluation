from typing import List

class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        # Group forbidden words by their lengths for faster checks
        forb_by_len = {}
        for f in forbidden:
            forb_by_len.setdefault(len(f), set()).add(f)
        
        n = len(word)
        # last_bad_pos is the latest start index of a forbidden substring ending at or before current i
        last_bad_pos = -1
        ans = 0
        
        # We only need to consider forbidden lengths up to 10
        max_forb_len = 10
        
        for i in range(n):
            # Check all substrings ending at i of lengths up to max_forb_len
            # If any of these substrings is forbidden, move the last_bad_pos
            for L in range(1, min(max_forb_len, i + 1) + 1):
                if L in forb_by_len:
                    start = i - L + 1
                    # slice word[start:i+1]
                    if word[start:i+1] in forb_by_len[L]:
                        # this substring invalidates any window that includes it
                        last_bad_pos = max(last_bad_pos, start)
            # Current valid window is (last_bad_pos+1 ... i)
            ans = max(ans, i - last_bad_pos)
        
        return ans