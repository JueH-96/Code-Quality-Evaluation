import collections

class Solution:
    def minimumLength(self, s: str) -> int:
        counter = collections.Counter(s)
        min_len = 0
        for count in counter.values():
            if count % 2 == 1:
                min_len += 1
            else:
                min_len += 2
        return min_len