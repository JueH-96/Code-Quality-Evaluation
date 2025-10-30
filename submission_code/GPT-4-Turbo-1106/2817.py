class Solution:
    def minimumCost(self, s: str) -> int:
        n = len(s)
        prefix_ones = [0] * (n + 1)
        suffix_zeros = [0] * (n + 1)
        
        # Calculate prefix sum of ones
        for i in range(1, n + 1):
            prefix_ones[i] = prefix_ones[i - 1] + (s[i - 1] == '1')
        
        # Calculate suffix sum of zeros
        for i in range(n - 1, -1, -1):
            suffix_zeros[i] = suffix_zeros[i + 1] + (s[i] == '0')
        
        # Calculate the minimum cost
        min_cost = float('inf')
        for i in range(n + 1):
            # Cost of making all characters '0' up to index i-1 and all '1' from index i
            cost = prefix_ones[i] + suffix_zeros[i]
            min_cost = min(min_cost, cost)
        
        return min_cost