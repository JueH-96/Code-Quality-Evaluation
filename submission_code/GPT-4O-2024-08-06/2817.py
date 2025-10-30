class Solution:
    def minimumCost(self, s: str) -> int:
        n = len(s)
        cost = 0
        
        # We will calculate the cost to make all characters '0' or all '1'
        # and take the minimum of both.
        
        # Cost to make all characters '0'
        cost_to_zeros = 0
        for i in range(1, n):
            if s[i] != s[i - 1]:
                # If there's a change, we need to flip from the current position
                cost_to_zeros += min(i, n - i)
        
        # Cost to make all characters '1'
        cost_to_ones = 0
        for i in range(1, n):
            if s[i] != s[i - 1]:
                # If there's a change, we need to flip from the current position
                cost_to_ones += min(i, n - i)
        
        # The minimum cost to make all characters equal
        cost = min(cost_to_zeros, cost_to_ones)
        
        return cost