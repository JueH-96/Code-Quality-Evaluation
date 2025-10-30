class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        freq = [0] * 26
        for c in word:
            freq[ord(c) - ord('a')] += 1
        max_freq = max(freq)
        min_del = len(word)  # Initialize with maximum possible deletions
        
        for m in range(0, max_freq + 1):
            total = 0
            for f in freq:
                if f == 0:
                    continue
                if f > m:
                    total += f - m
                elif f < (m - k):
                    total += f
            if total < min_del:
                min_del = total
        
        return min_del