from typing import List

class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        forb = set(forbidden)
        left = 0
        ans = 0
        n = len(word)
        
        for i in range(n):
            # Check all substrings ending at i up to length 10
            for l in range(1, min(10, i + 1) + 1):
                if word[i - l + 1 : i + 1] in forb:
                    # Move left past the start of this forbidden substring
                    left = max(left, i - l + 2)
            # Update the max valid substring length ending at i
            ans = max(ans, i - left + 1)
        
        return ans