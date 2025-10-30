class Solution:
    def minimumCost(self, s: str) -> int:
        n = len(s)
        total_cost = 0
        # We add the minimum cost for each boundary change
        # If s[i] != s[i+1], we must flip across that boundary;
        # it costs min(i+1, n-(i+1)) to do so with a single prefix or suffix flip.
        for i in range(n - 1):
            if s[i] != s[i + 1]:
                total_cost += min(i + 1, n - (i + 1))
        return total_cost