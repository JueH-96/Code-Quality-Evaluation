class Solution:
    def minOperations(self, k: int) -> int:
        # Start with the initial array sum
        current_sum = 1
        # Start with the initial array length
        current_length = 1
        # Initialize the number of operations
        operations = 0
        
        # While the current sum is less than k
        while current_sum < k:
            # If duplicating the current array would help reach or exceed k
            if current_sum * 2 <= k:
                # Duplicate the array
                current_sum *= 2
                current_length *= 2
            else:
                # Otherwise, increase the smallest element (1) to reach k
                increase = k - current_sum
                current_sum += increase
                operations += increase
                break
            # Increment the operation count for duplication
            operations += 1
        
        return operations