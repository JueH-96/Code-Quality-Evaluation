class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        n = len(word)
        count = {}
        max_freq = 0
        for i in range(0, n, k):
            substring = word[i:i+k]
            count[substring] = count.get(substring, 0) + 1
            if count[substring] > max_freq:
                max_freq = count[substring]
        total_blocks = n // k
        return total_blocks - max_freq