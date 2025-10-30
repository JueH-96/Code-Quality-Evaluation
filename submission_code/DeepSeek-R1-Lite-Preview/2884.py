from typing import List

class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        forbidden_set = set(forbidden)
        if forbidden:
            max_length_forbidden = max(len(s) for s in forbidden)
        else:
            max_length_forbidden = 0
        left = 0
        max_len = 0
        n = len(word)
        
        for right in range(n):
            for l in range(1, max_length_forbidden + 1):
                if right - l + 1 < 0:
                    continue
                substring = word[right - l + 1:right + 1]
                if substring in forbidden_set:
                    left = max(left, right - l + 2)
            current_window_length = right - left + 1
            max_len = max(max_len, current_window_length)
        
        return max_len