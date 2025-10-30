class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        n = len(word)
        # Split the word into n/k parts, each of length k
        parts = [word[i:i+k] for i in range(0, n, k)]
        # Count the frequency of each part
        freq = {}
        for part in parts:
            if part in freq:
                freq[part] += 1
            else:
                freq[part] = 1
        # Find the part with the maximum frequency
        max_freq = max(freq.values())
        # The minimum number of operations is the total number of parts minus the maximum frequency
        return len(parts) - max_freq