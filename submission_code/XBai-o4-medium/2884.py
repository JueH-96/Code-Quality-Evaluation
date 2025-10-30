from typing import List

class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        forbidden_set = set(forbidden)
        max_len_forbidden = max(len(f) for f in forbidden) if forbidden else 0
        n = len(word)
        left = 0
        max_length = 0
        
        for right in range(n):
            # Check all possible substrings ending at 'right' with length 1 to max_len_forbidden
            for l in range(1, max_len_forbidden + 1):
                start = right - l + 1
                if start < 0:
                    break
                substring = word[start:right+1]
                if substring in forbidden_set:
                    left = max(left, start + 1)
            current_length = right - left + 1
            if current_length > max_length:
                max_length = current_length
        
        return max_length