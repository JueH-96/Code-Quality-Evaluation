class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        n = len(word)
        freq = [0] * 26
        for i in range(k):
            freq[ord(word[i]) - ord('a')] += 1
        operations = sum(freq) - max(freq)
        for i in range(k, n):
            freq[ord(word[i]) - ord('a')] += 1
            freq[ord(word[i - k]) - ord('a')] -= 1
            operations = min(operations, sum(freq) - max(freq))
        return operations