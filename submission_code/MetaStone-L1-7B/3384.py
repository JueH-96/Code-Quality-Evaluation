from collections import defaultdict

class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        substrings = defaultdict(int)
        m = len(word) // k
        for i in range(m):
            substr = word[i*k : (i+1)*k]
            substrings[substr] += 1
        if not substrings:
            return 0
        max_freq = max(substrings.values())
        if max_freq == len(substrings):
            return 0
        else:
            return len(substrings) - max_freq