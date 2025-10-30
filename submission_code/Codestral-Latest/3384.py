class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        n = len(word)
        operations = 0

        for i in range(0, n - k, k):
            if word[i:i + k] != word[i + k:i + 2 * k]:
                operations += 1

        return operations