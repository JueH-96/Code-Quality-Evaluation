class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        from collections import defaultdict
        m = len(word) // k
        freq = defaultdict(int)
        for i in range(0, len(word), k):
            block = word[i:i+k]
            freq[block] += 1
        max_freq = max(freq.values())
        return m - max_freq