class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        from collections import Counter
        from heapq import heapify, heappop

        counter = Counter(word)
        freqs = [-freq for freq in counter.values()]
        heapify(freqs)
        deletions = 0

        while len(freqs) > 1:
            min_freq = heappop(freqs)
            if min_freq >= freqs[0] + k:
                deletions += min_freq - (freqs[0] + k - 1)
                min_freq = freqs[0] + k - 1
            if min_freq < 0:
                heappush(freqs, min_freq)

        return deletions