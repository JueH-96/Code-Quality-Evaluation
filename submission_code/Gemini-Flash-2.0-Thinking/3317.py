from collections import Counter
from typing import List

class Solution:
    def maxPalindromesAfterOperations(self, words: List[str]) -> int:
        char_counts = Counter()
        for word in words:
            char_counts.update(word)

        words.sort(key=len)

        palindrome_count = 0
        available_pairs = sum(count // 2 for count in char_counts.values())
        available_odds = sum(count % 2 for count in char_counts.values())

        for word in words:
            L = len(word)
            needed_pairs = L // 2

            if L % 2 == 0:
                if available_pairs >= needed_pairs:
                    palindrome_count += 1
                    available_pairs -= needed_pairs
            else:
                if available_pairs >= needed_pairs and available_odds >= 1:
                    palindrome_count += 1
                    available_pairs -= needed_pairs
                    available_odds -= 1

        return palindrome_count