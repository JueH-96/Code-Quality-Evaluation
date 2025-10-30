from collections import Counter

class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        total_blocks = len(word) // k
        counts = Counter()
        for i in range(0, len(word), k):
            block = word[i:i+k]
            counts[block] += 1
        max_freq = max(counts.values())
        return total_blocks - max_freq