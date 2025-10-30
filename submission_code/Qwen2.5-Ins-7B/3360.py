class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        from collections import Counter
        
        freq = sorted(Counter(word).values())
        n = len(freq)
        res = float('inf')
        
        for i in range(n):
            delete = 0
            for j in range(i):
                delete += freq[j]
            for j in range(i, n):
                delete += max(0, freq[j] - freq[i] - k)
            res = min(res, delete)
        
        return res