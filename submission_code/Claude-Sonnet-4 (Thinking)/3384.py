class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        n = len(word)
        
        # Count frequency of each substring of length k at positions 0, k, 2k, ...
        freq = {}
        for i in range(0, n, k):
            substr = word[i:i+k]
            freq[substr] = freq.get(substr, 0) + 1
        
        # The minimum operations = total_chunks - max_frequency
        total_chunks = n // k
        max_freq = max(freq.values())
        
        return total_chunks - max_freq