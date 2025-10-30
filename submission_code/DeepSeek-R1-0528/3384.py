class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        m = len(word) // k
        segment_counts = {}
        for i in range(0, len(word), k):
            segment = word[i:i+k]
            segment_counts[segment] = segment_counts.get(segment, 0) + 1
        max_freq = max(segment_counts.values())
        return m - max_freq