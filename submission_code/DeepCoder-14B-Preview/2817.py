class Solution:
    def minimumCost(self, s: str) -> int:
        n = len(s)
        total = 0
        for i in range(n - 1):
            if s[i] != s[i + 1]:
                cost_prefix = i + 1
                cost_suffix = n - (i + 1)
                total += min(cost_prefix, cost_suffix)
        return total