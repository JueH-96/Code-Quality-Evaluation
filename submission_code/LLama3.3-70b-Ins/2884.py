from typing import List

class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        def is_valid(substring: str) -> bool:
            """Check if a substring is valid."""
            for f in forbidden:
                if f in substring:
                    return False
            return True

        max_length = 0
        for i in range(len(word)):
            for j in range(i + 1, len(word) + 1):
                substring = word[i:j]
                if is_valid(substring):
                    max_length = max(max_length, len(substring))
        return max_length