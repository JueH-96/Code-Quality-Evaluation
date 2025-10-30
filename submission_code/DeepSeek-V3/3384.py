class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        n = len(word)
        # Split the word into n/k segments of length k
        segments = [word[i:i+k] for i in range(0, n, k)]
        # Count the frequency of each segment
        from collections import defaultdict
        freq = defaultdict(int)
        for seg in segments:
            freq[seg] += 1
        # Find the segment with the maximum frequency
        max_freq = max(freq.values())
        # The minimum number of operations is the total number of segments minus the frequency of the most common segment
        return len(segments) - max_freq