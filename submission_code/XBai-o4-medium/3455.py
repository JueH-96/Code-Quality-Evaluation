from collections import Counter

class Solution:
    def minimumLength(self, s: str) -> int:
        cnt = Counter(s)
        res = 0
        for char, freq in cnt.items():
            if freq == 1:
                res += 1
            else:
                if freq % 2 == 1:
                    res += 1
                else:
                    res += 2
        return res