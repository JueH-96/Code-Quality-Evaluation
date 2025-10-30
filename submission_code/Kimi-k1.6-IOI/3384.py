from collections import Counter

class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        n = len(word)
        num_blocks = n // k
        blocks = [word[i*k : (i+1)*k] for i in range(num_blocks)]
        counts = Counter(blocks)
        max_freq = max(counts.values())
        return num_blocks - max_freq