class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        n = len(word)
        m = n // k  # number of segments

        freq = {}
        # Count frequency of each segment of length k
        for i in range(m):
            segment = word[i*k:(i+1)*k]
            freq[segment] = freq.get(segment, 0) + 1

        # Find the maximum frequency of any segment
        max_freq = max(freq.values())

        # Number of operations is total segments minus most frequent segment
        return m - max_freq