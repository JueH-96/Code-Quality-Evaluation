class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        from collections import defaultdict
        n = len(word)
        m = n // k  # Number of blocks
        # Constants for hashing
        base1 = 911
        mod1 = 10**9 + 7
        base2 = 3571
        mod2 = 10**9 + 9

        counts = defaultdict(int)
        max_count = 0

        for i in range(0, n, k):
            s = word[i : i + k]
            h1 = 0
            h2 = 0
            for c in s:
                num = ord(c) - ord('a') + 1
                h1 = (h1 * base1 + num) % mod1
                h2 = (h2 * base2 + num) % mod2
            hkey = (h1, h2)
            counts[hkey] += 1
            if counts[hkey] > max_count:
                max_count = counts[hkey]

        min_operations = m - max_count
        return min_operations