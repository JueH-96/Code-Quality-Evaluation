class Solution:
    def minimumCost(self, s: str) -> int:
        n = len(s)
        total_cost = 0
        for i in range(n - 1):
            if s[i] != s[i + 1]:
                total_cost += min(i + 1, n - i - 1)
        return total_cost