class Solution:
    def minOperations(self, k: int) -> int:
        if k <= 1:
            return 0
        
        min_ops = k - 1  # Worst case: just increase the single element to k
        
        # Try different numbers of elements (copies)
        for num_elements in range(1, k + 1):
            # Calculate the value each element should have
            value_needed = (k + num_elements - 1) // num_elements  # This is ceil(k/num_elements)
            
            # Operations needed:
            # - Increase from 1 to value_needed: (value_needed - 1) operations
            # - Create (num_elements - 1) duplicates: (num_elements - 1) operations
            operations = (value_needed - 1) + (num_elements - 1)
            
            min_ops = min(min_ops, operations)
            
            # Early termination: if value_needed is 1, we won't get better results
            if value_needed == 1:
                break
        
        return min_ops