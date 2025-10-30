class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        n = len(word)
        m = n // k
        substrings = [word[i * k : i * k + k] for i in range(m)]

        frequencies = {}
        for sub in substrings:
            frequencies[sub] = frequencies.get(sub, 0) + 1

        if not frequencies:
            return 0

        max_frequency = 0
        for freq in frequencies.values():
            max_frequency = max(max_frequency, freq)

        min_ops = m - max_frequency
        return min_ops