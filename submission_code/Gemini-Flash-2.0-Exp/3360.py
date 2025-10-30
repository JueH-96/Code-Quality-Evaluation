class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        from collections import Counter
        counts = Counter(word)
        frequencies = list(counts.values())
        n = len(word)
        ans = n
        for target_freq in range(1, n + 1):
            deletions = 0
            for freq in frequencies:
                if abs(freq - target_freq) > k:
                    deletions += min(freq, abs(freq - target_freq) - k)
            ans = min(ans, deletions)
        return ans