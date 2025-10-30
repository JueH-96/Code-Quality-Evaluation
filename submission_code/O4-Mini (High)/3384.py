class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        n = len(word)
        # number of k-sized blocks
        m = n // k
        
        # count occurrences of each block
        freq = {}
        for i in range(0, n, k):
            block = word[i:i+k]
            freq[block] = freq.get(block, 0) + 1
        
        # the best we can do is pick the block string that appears most often,
        # and copy it to all other blocks
        max_count = max(freq.values())
        return m - max_count