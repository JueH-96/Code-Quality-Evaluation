class Solution:
    def minimumCost(self, s: str) -> int:
        n = len(s)
        if n == 1:
            return 0
        
        # Cost to make all characters '0'
        cost_to_zero = 0
        # Cost to make all characters '1'
        cost_to_one = 0
        
        # Calculate initial costs assuming the whole string is flipped to '0' or '1'
        for i in range(n):
            if s[i] == '1':
                cost_to_zero += (n - i)  # Cost to flip this '1' to '0' using second operation
            else:
                cost_to_one += (i + 1)  # Cost to flip this '0' to '1' using first operation
        
        # Minimum cost initialized to the minimum of converting all to '0' or all to '1'
        min_cost = min(cost_to_zero, cost_to_one)
        
        # Prefix and suffix costs to flip to '0' or '1'
        prefix_cost_to_zero = 0
        suffix_cost_to_one = 0
        
        # Calculate prefix costs for converting to '0' and suffix costs for converting to '1'
        for i in range(n):
            if s[i] == '1':
                prefix_cost_to_zero += (i + 1)
            if s[n - i - 1] == '0':
                suffix_cost_to_one += (i + 1)
            
            # Update minimum cost considering the current split at i
            if i < n - 1:
                min_cost = min(min_cost, prefix_cost_to_zero + suffix_cost_to_one)
        
        return min_cost