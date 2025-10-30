from collections import Counter
from typing import List

class Solution:
    def longestCommonPrefix(self, words: List[str], k: int) -> List[int]:
        n = len(words)
        freq = Counter(words)
        answer = [0] * n
        
        for i in range(n):
            # Remove the word at index i temporarily
            word = words[i]
            freq[word] -= 1
            
            # Find the longest string that appears at least k times
            max_length = 0
            for w, count in freq.items():
                if count >= k:
                    max_length = max(max_length, len(w))
            
            answer[i] = max_length
            
            # Restore the word count
            freq[word] += 1
        
        return answer