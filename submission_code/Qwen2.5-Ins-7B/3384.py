class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        n = len(word)
        prefix = [0] * (n + 1)
        for i in range(k, n, k):
            prefix[i] = prefix[i - k] + (word[i - k] != word[i])
        return prefix[-1]