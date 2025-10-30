class Solution:
    def minOperations(self, s1: str, s2: str, x: int) -> int:
        diffs = []
        n = len(s1)
        for i in range(n):
            if s1[i] != s2[i]:
                diffs.append(i)
        m = len(diffs)
        if m % 2 != 0:
            return -1
        if m == 0:
            return 0
        diffs.sort()
        res = 0
        for i in range(0, m, 2):
            d = diffs[i+1] - diffs[i]
            res += min(d, x)
        return res