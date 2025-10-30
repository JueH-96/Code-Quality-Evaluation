class Solution:
    def minimumLength(self, s: str) -> int:
        from collections import Counter
        freq = Counter(s)
        min_length = 0
        for count in freq.values():
            if count == 1:
                min_length += 1
            elif count == 2:
                min_length += 2
            else:
                if count % 2 == 0:
                    min_length += 2
                else:
                    min_length += 1
        return min_length