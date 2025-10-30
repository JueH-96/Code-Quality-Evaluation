from typing import List
from collections import Counter

class Solution:
    def maxPalindromesAfterOperations(self, words: List[str]) -> int:
        # Count total available pairs from all characters
        cnt = Counter()
        for w in words:
            cnt.update(w)
        total_pairs = sum(v // 2 for v in cnt.values())

        # Each word of length L needs floor(L/2) pairs to become a palindrome
        need = [len(w) // 2 for w in words]
        need.sort()

        # Greedily take as many words (with smallest pair‐needs) as we can afford
        used = 0
        res = 0
        for p in need:
            if used + p <= total_pairs:
                used += p
                res += 1
            else:
                break

        return res