from collections import Counter

class Solution:
    def minAnagramLength(self, s: str) -> int:
        n = len(s)
        for length in range(1, n+1):
            if n % length == 0:
                target_count = n // length
                target_freq = Counter(s[:length])
                if all(target_freq[c] == target_count for c in target_freq):
                    return length
        return n