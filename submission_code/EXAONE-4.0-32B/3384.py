from collections import Counter

class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        n = len(word)
        m = n // k
        blocks = [word[i*k:(i+1)*k] for i in range(m)]
        cnt = Counter(blocks)
        max_freq = cnt.most_common(1)[0][1]
        return m - max_freq