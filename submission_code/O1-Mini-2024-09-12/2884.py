from typing import List

class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        forbidden_set = set(forbidden)
        max_forbidden_length = max((len(s) for s in forbidden), default=0)
        start = 0
        max_len = 0

        for end in range(len(word)):
            # Check substrings ending at 'end' with length up to max_forbidden_length
            check_start = max(start, end - max_forbidden_length + 1)
            for i in range(check_start, end + 1):
                substring = word[i:end + 1]
                if substring in forbidden_set:
                    start = max(start, i + 1)
                    break  # No need to check longer substrings
            max_len = max(max_len, end - start + 1)
        
        return max_len