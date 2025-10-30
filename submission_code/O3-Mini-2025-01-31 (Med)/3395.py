from collections import Counter

class Solution:
    def minAnagramLength(self, s: str) -> int:
        n = len(s)
        freq = Counter(s)
        # We need to find the smallest m (length of t) such that:
        #   if k = n // m, then for each letter, freq(letter) is a multiple of k.
        # Since s is a concatenation of k anagrams of t, each block t (in some order)
        # contributes evenly to the total counts.
        for m in range(1, n + 1):
            if n % m != 0:
                continue
            k = n // m
            valid = True
            for count in freq.values():
                if count % k != 0:
                    valid = False
                    break
            if valid:
                return m
        return n  # Fallback, though this line is normally unreachable.