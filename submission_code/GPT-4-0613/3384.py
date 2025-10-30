class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        n = len(word)
        freq = [0]*26
        max_freq = 0
        operations = 0
        for i in range(k):
            freq = [0]*26
            j = i
            while j < n:
                freq[ord(word[j]) - ord('a')] += 1
                j += k
            max_freq = max(freq)
            operations += (j//k - max_freq)
        return operations