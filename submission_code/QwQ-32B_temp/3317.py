from collections import Counter
from typing import List

class Solution:
    def maxPalindromesAfterOperations(self, words: List[str]) -> int:
        count = Counter()
        for word in words:
            count.update(word)
        
        total_pairs = 0
        original_singles = 0
        for c in count:
            cnt = count[c]
            total_pairs += cnt // 2
            original_singles += cnt % 2
        
        # Sort words by their length in ascending order
        words_sorted = sorted(words, key=lambda x: len(x))
        
        current_sum = 0
        current_K = 0
        res = 0
        
        for word in words_sorted:
            l = len(word)
            pairs_needed = l // 2
            new_sum = current_sum + pairs_needed
            new_K = current_K + (1 if l % 2 else 0)
            
            if new_sum > total_pairs:
                continue
            
            required = new_K
            available = original_singles + 2 * (total_pairs - new_sum)
            if required <= available:
                current_sum = new_sum
                current_K = new_K
                res += 1
        
        return res