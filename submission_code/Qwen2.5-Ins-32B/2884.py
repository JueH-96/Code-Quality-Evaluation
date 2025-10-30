from typing import List

class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        forbidden_set = set(forbidden)
        max_len = 0
        start = 0
        
        for end in range(len(word)):
            for i in range(end, max(end - 10, start - 1), -1):
                if word[i:end + 1] in forbidden_set:
                    start = i + 1
                    break
            max_len = max(max_len, end - start + 1)
        
        return max_len