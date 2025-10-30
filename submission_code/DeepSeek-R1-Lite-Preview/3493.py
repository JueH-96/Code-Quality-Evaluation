class Solution:
    def maxOperations(self, s: str) -> int:
        n = len(s)
        ones_positions = []
        
        # Collect positions of '1's
        for i in range(n):
            if s[i] == '1':
                ones_positions.append(i)
        
        total_operations = 0
        k = len(ones_positions)
        
        # Calculate operations for each '1'
        for i in range(k - 1):
            zeros_between = ones_positions[i + 1] - ones_positions[i] - 1
            total_operations += zeros_between
        
        # Add '0's after the last '1'
        if k > 0:
            zeros_after_last_one = n - ones_positions[-1] - 1
            total_operations += zeros_after_last_one
        
        return total_operations