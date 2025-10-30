class Solution:
    def minimumCost(self, s: str) -> int:
        n = len(s)
        res = 0
        for i in range(n - 1):
            if s[i] != s[i + 1]:
                res += min(i + 1, n - (i + 1))
        return res