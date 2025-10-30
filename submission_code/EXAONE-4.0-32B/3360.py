class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        freq = [0] * 26
        for ch in word:
            idx = ord(ch) - ord('a')
            freq[idx] += 1
        
        maxF = max(freq)
        min_deletions = float('inf')
        
        for L in range(0, maxF + 1):
            deletions = 0
            for f in freq:
                if f < L:
                    deletions += f
                elif f > L + k:
                    deletions += f - (L + k)
            if deletions < min_deletions:
                min_deletions = deletions
        
        return min_deletions