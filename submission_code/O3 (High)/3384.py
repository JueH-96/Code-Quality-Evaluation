class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        n = len(word)
        block_count = n // k                   # how many k-length blocks the word contains
        
        freq = {}                              # frequency of each distinct k-length block
        for i in range(0, n, k):
            block = word[i:i + k]
            freq[block] = freq.get(block, 0) + 1
        
        max_same = max(freq.values())          # largest number of identical blocks already present
        return block_count - max_same          # all other blocks must be overwritten, 1 per operation