class Solution:
    def minimumCost(self, s: str) -> int:
        n = len(s)
        if n == 1:
            return 0
        
        # Determine the target character (either all '0' or all '1')
        # We need to calculate the cost for both scenarios and choose the minimum
        
        # Function to calculate the cost to make all characters equal to target
        def calculate_cost(target):
            cost = 0
            # Traverse the string and apply operations as needed
            for i in range(n):
                if s[i] != target:
                    # Apply the first operation: invert from 0 to i
                    cost += i + 1
                    # Update the target for the next characters
                    target = '1' if target == '0' else '0'
            return cost
        
        # Calculate the cost for both targets
        cost_all_0 = calculate_cost('0')
        cost_all_1 = calculate_cost('1')
        
        # Return the minimum cost
        return min(cost_all_0, cost_all_1)