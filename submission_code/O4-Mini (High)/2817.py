class Solution:
    def minimumCost(self, s: str) -> int:
        n = len(s)
        cost = 0
        # For every boundary where s[i] != s[i+1],
        # we must eliminate that flip at least once,
        # either by a prefix flip of cost (i+1) or a suffix flip of cost (n-i-1).
        # The cheapest way to eliminate this boundary is min(i+1, n-i-1).
        for i in range(n - 1):
            if s[i] != s[i + 1]:
                cost += min(i + 1, n - i - 1)
        return cost