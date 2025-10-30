class Solution:
    def minOperations(self, k: int) -> int:
        if k == 1:
            return 0
        
        operations = 0
        current_sum = 1
        max_element = 1
        
        while current_sum < k:
            if current_sum + max_element <= k:
                # Duplicate the largest element
                current_sum += max_element
                operations += 1
            else:
                # Calculate how much we need to add to reach or exceed k
                needed = k - current_sum
                if needed < max_element:
                    # Increment the largest element to make up the difference
                    operations += needed
                    current_sum += needed
                else:
                    # Increment the largest element fully
                    operations += max_element
                    current_sum += max_element
                    max_element *= 2  # After incrementing fully, it's like duplicating it
            
        return operations