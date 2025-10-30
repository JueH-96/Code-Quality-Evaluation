from collections import defaultdict

class Solution:
    def maximumLength(self, s: str) -> int:
        n = len(s)
        for l in range(n, 0, -1):
            counts = defaultdict(int)
            for i in range(n - l + 1):
                substr = s[i:i+l]
                if len(set(substr)) == 1:
                    counts[substr] += 1
            for cnt in counts.values():
                if cnt >= 3:
                    return l
        return -1