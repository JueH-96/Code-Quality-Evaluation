from typing import List

class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        forbidden_set = set(forbidden)
        max_len = 0
        start = 0
        n = len(word)
        
        for i in range(n):
            # Check up to 10 last characters (since forbidden[i] length <= 10)
            limit = min(10, i + 1)
            for length in range(1, limit + 1):
                # Substring ending at i with length = length
                segment = word[i - length + 1 : i + 1]
                if segment in forbidden_set:
                    # Move start to exclude this forbidden substring
                    start = max(start, i - length + 2)
            # Update the maximum valid substring length
            max_len = max(max_len, i - start + 1)
        
        return max_len