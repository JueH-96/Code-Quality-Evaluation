class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        n = len(word)
        count = {}
        for i in range(0, n, k):
            substring = word[i:i + k]
            if substring not in count:
                count[substring] = 0
            count[substring] += 1
        return len(count) -1 if len(count)>0 else 0