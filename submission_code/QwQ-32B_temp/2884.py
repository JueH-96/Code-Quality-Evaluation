from typing import List

class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        forbidden_set = set(forbidden)
        max_len = 0
        start = 0
        n = len(word)
        
        for end in range(n):
            current_max_start = start
            # Check all possible substring lengths from 1 to 10 ending at 'end'
            for l in range(1, 11):
                start_sub = end - l + 1
                if start_sub < 0:
                    continue
                substring = word[start_sub:end+1]
                if substring in forbidden_set:
                    candidate = end - l + 2
                    if candidate > current_max_start:
                        current_max_start = candidate
            # Update the start to the maximum required position
            start = current_max_start
            current_length = end - start + 1
            if current_length > max_len:
                max_len = current_length
        
        return max_len