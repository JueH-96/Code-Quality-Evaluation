import bisect

class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        from collections import Counter
        freq = list(Counter(word).values())
        freq.sort()
        n = len(freq)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + freq[i]
        max_sum = 0
        for i in range(n):
            m = freq[i]
            t = m - k
            l = bisect.bisect_left(freq, t)
            current_sum = prefix[i + 1] - prefix[l]
            current_sum += (n - i - 1) * m
            if current_sum > max_sum:
                max_sum = current_sum
        return len(word) - max_sum