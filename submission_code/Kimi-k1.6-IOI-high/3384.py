from collections import Counter

class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        chunks = [word[i:i+k] for i in range(0, len(word), k)]
        freq = Counter(chunks)
        max_count = max(freq.values())
        return len(chunks) - max_count