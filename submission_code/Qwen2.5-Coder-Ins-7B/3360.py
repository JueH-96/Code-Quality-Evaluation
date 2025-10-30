class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        from collections import Counter
        freq = Counter(word)
        freq = sorted(freq.values())
        n = len(freq)
        ans = float('inf')
        for i in range(n):
            max_freq = freq[i] + k
            if max_freq < freq[0]:
                continue
            deletions = 0
            for j in range(n):
                if freq[j] > max_freq:
                    deletions += freq[j] - max_freq
            ans = min(ans, deletions)
        return ans