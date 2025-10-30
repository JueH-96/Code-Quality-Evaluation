from collections import defaultdict
from typing import List

class Solution:
    def maxPalindromesAfterOperations(self, words: List[str]) -> int:
        freq = defaultdict(int)
        for word in words:
            for c in word:
                freq[c] += 1
        
        P = 0
        for count in freq.values():
            P += count // 2
        
        arr = [len(word) // 2 for word in words]
        arr.sort()
        
        count = 0
        current_sum = 0
        for num in arr:
            if current_sum + num <= P:
                current_sum += num
                count += 1
            else:
                break
        return count