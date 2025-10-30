from collections import Counter

class Solution:
    def minimumLength(self, s: str) -> int:
        cnt = Counter(s)
        res = 0
        for k in cnt.values():
            if k % 2 == 1:
                # odd count → can reduce to 1
                res += 1
            else:
                # even count → can reduce to 2 (if k>=2)
                res += 2
        return res