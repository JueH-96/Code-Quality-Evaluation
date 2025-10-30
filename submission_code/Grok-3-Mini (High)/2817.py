class Solution:
    def minimumCost(self, s: str) -> int:
        n = len(s)
        cost = 0
        for j in range(n - 1):
            if s[j] != s[j + 1]:
                cost += min(j + 1, n - j - 1)
        return cost