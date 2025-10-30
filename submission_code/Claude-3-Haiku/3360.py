class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        n = len(word)
        freq = [0] * 26
        for c in word:
            freq[ord(c) - ord('a')] += 1
        
        max_diff = max(freq) - min(freq)
        if max_diff <= k:
            return n - max_diff
        
        min_deletions = n
        for i in range(26):
            if freq[i] > 0:
                min_deletions = min(min_deletions, n - freq[i])
        
        return min_deletions