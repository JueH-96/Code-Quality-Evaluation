class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        from collections import defaultdict
        freq = defaultdict(int)
        for c in s:
            freq[c] += 1
        K = max(freq.values()) if freq else 0
        count = defaultdict(int)
        res = []
        for c in s:
            count[c] += 1
            if count[c] > K - 1:
                res.append(c)
        return ''.join(res)