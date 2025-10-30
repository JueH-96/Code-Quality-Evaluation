from collections import Counter

class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        counter = Counter(word)
        freq = sorted(counter.values())
        l, r = 0, len(freq) - 1
        res = len(freq)
        while l <= r:
            if freq[r] - freq[l] <= k:
                res = min(res, r - l + 1)
                l += 1
            else:
                r -= 1
        return len(word) - res