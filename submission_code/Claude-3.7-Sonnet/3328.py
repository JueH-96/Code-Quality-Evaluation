class Solution:
    def minOperations(self, k: int) -> int:
        if k <= 1:
            return 0  # Initial array sum already meets or exceeds k
        
        min_operations = float('inf')
        sqrt_k = int(k ** 0.5)
        
        # Check values from 1 to 2 * sqrt(k) to ensure we find the minimum
        for x in range(1, 2 * sqrt_k + 1):
            n = (k + x - 1) // x  # Ceiling division
            operations = (x - 1) + (n - 1)
            min_operations = min(min_operations, operations)
            
        return min_operations