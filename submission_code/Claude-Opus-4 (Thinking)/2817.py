class Solution:
    def minimumCost(self, s: str) -> int:
        n = len(s)
        total_cost = 0
        
        # Find all transitions and calculate minimum cost to fix each
        for i in range(n - 1):
            if s[i] != s[i + 1]:
                # Cost to flip left part (0 to i)
                cost_left = i + 1
                # Cost to flip right part (i+1 to n-1)
                cost_right = n - i - 1
                # Choose the cheaper option
                total_cost += min(cost_left, cost_right)
        
        return total_cost