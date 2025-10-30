class Solution:
    def minimumCost(self, s: str) -> int:
        n = len(s)
        total_cost = 0
        
        # Check each adjacent pair
        for i in range(n - 1):
            if s[i] != s[i + 1]:
                # We need to make s[i] and s[i+1] equal
                # Option 1: Use operation 1 at index i (cost = i + 1)
                # Option 2: Use operation 2 at index i + 1 (cost = n - i - 1)
                total_cost += min(i + 1, n - i - 1)
        
        return total_cost