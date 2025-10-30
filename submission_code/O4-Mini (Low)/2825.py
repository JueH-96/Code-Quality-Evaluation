class Solution:
    def minimizedStringLength(self, s: str) -> int:
        from collections import Counter
        cnt = Counter(s)
        # For each character with frequency k, the minimal survivors on a path
        # under the given operations is the size of a minimum dominating set
        # on a path of length k, which is ceil(k / 3).
        res = 0
        for k in cnt.values():
            res += (k + 2) // 3
        return res