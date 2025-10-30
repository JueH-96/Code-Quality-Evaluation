from collections import Counter

class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        chunks = [word[i:i+k] for i in range(0, len(word), k)]
        counts = Counter(chunks)
        max_count = max(counts.values())
        return len(chunks) - max_count