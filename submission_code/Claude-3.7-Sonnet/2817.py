class Solution:
    def minimumCost(self, s: str) -> int:
        n = len(s)
        total_cost = 0
        
        for i in range(n - 1):
            if s[i] != s[i + 1]:
                # If adjacent characters differ, we need to make them the same
                # We can either invert from 0 to i, or from i+1 to n-1
                # Choose the option with minimum cost
                cost = min(i + 1, n - (i + 1))
                total_cost += cost
        
        return total_cost