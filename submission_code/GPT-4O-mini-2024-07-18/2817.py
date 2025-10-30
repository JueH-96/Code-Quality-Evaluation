class Solution:
    def minimumCost(self, s: str) -> int:
        n = len(s)
        
        # Cost to make all '0's
        cost_to_all_0 = [0] * (n + 1)
        for i in range(n):
            cost_to_all_0[i + 1] = cost_to_all_0[i] + (s[i] == '1')
        
        # Cost to make all '1's
        cost_to_all_1 = [0] * (n + 1)
        for i in range(n):
            cost_to_all_1[i + 1] = cost_to_all_1[i] + (s[i] == '0')
        
        min_cost = float('inf')
        
        # Calculate minimum cost
        for i in range(n + 1):
            cost = cost_to_all_0[i] + (n - i) - cost_to_all_1[n] + cost_to_all_1[i]
            min_cost = min(min_cost, cost)
        
        return min_cost