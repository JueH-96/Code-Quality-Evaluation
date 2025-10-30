class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        n = len(word)
        if k % n == 0:
            return 0
        else:
            return (k // n) * n + (k % n)