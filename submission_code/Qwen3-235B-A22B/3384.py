from collections import Counter

class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        n = len(word)
        blocks = [word[i:i+k] for i in range(0, n, k)]
        count = Counter(blocks)
        max_freq = max(count.values(), default=0)
        return len(blocks) - max_freq