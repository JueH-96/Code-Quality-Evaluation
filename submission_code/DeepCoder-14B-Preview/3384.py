from collections import defaultdict

class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        n = len(word)
        m = n // k
        blocks = [word[i*k : (i+1)*k] for i in range(m)]
        
        freq = defaultdict(int)
        for b in blocks:
            freq[b] += 1
        
        max_freq = max(freq.values()) if freq else 0
        return m - max_freq