class Solution:
    def minOperations(self, k: int) -> int:
        if k == 1:
            return 0
        
        operations = 0
        current_sum = 1
        current_value = 1
        
        while current_sum < k:
            # Calculate the optimal number of duplications for the current value
            duplications = (k - current_sum + current_value - 1) // current_value - 1
            if duplications < 0:
                duplications = 0
            
            # Calculate the optimal number of increments for the current value
            increments = (k - current_sum) // (duplications + 1)
            if increments < 0:
                increments = 0
            
            # Choose the minimum number of operations between increments and duplications
            if increments + 1 <= duplications:
                current_value += increments + 1
                current_sum += increments + 1
                operations += increments + 1
            else:
                current_sum += current_value * (duplications + 1)
                operations += duplications + 1
        
        return operations