from typing import List

class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        if not forbidden:
            return len(word)
        
        forbidden_set = set(forbidden)
        max_len_forbidden = max(len(s) for s in forbidden)
        n = len(word)
        start = 0
        max_length = 0
        
        for j in range(n):
            current_start = start
            max_l = min(max_len_forbidden, j + 1)
            for l in range(1, max_l + 1):
                start_idx = j - l + 1
                if start_idx < 0:
                    continue
                substr = word[start_idx:j+1]
                if substr in forbidden_set:
                    current_start = max(current_start, start_idx + 1)
            start = current_start
            current_length = j - start + 1
            if current_length > max_length:
                max_length = current_length
        
        return max_length