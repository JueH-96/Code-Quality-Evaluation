from collections import defaultdict
from typing import List

class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        forbidden_sets = defaultdict(set)
        max_len = 0
        for s in forbidden:
            forbidden_sets[len(s)].add(s)
            max_len = max(max_len, len(s))
        
        if not forbidden_sets:
            return len(word)
        
        left = 0
        max_window = 0
        n = len(word)
        
        for i in range(n):
            # Check all possible lengths up to max_len and current possible
            for L in range(1, max_len + 1):
                if L > (i + 1):
                    continue  # Not enough characters to form length L
                substr_start = i - L + 1
                substr = word[substr_start:i + 1]
                if substr in forbidden_sets[L]:
                    left = max(left, substr_start + 1)
            
            current_length = i - left + 1
            if current_length > max_window:
                max_window = current_length
        
        return max_window