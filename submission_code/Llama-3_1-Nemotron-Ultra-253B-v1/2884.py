from typing import List

class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        forbidden_set = set(forbidden)
        max_len = 0
        start = 0
        for end in range(len(word)):
            for L in range(1, 11):
                substr_start = end - L + 1
                if substr_start < 0:
                    continue
                substring = word[substr_start:end+1]
                if substring in forbidden_set:
                    start = max(start, substr_start + 1)
            current_len = end - start + 1
            if current_len > max_len:
                max_len = current_len
        return max_len