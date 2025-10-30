class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        n = len(word)
        ans = 0
        for i in range(k):
            counts = {}
            for j in range(i, n, k):
                counts[word[j]] = counts.get(word[j], 0) + 1
            max_freq = 0
            for count in counts.values():
                max_freq = max(max_freq, count)
            ans += (n // k) - max_freq
        return ans