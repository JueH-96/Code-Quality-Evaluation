class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        n = len(word)
        groups = [word[i:i+k] for i in range(0, n, k)]
        from collections import Counter
        counter = Counter(groups)
        max_count = counter.most_common(1)[0][1]
        return len(groups) - max_count