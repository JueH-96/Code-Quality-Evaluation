class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        n = len(word)
        m = n // k  # Number of blocks
        blocks = [word[i*k : (i+1)*k] for i in range(m)]
        from collections import Counter
        count = Counter(blocks)
        max_freq = max(count.values())
        return m - max_freq