from typing import List

class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        # Initialize the count of operations
        operations = 0
        
        # Iterate over each number in the array
        for num in nums:
            # Calculate the remainder when the number is divided by 3
            remainder = num % 3
            
            # If the remainder is not 0, we need to perform operations to make it divisible by 3
            if remainder != 0:
                # The minimum number of operations required is the minimum between the remainder and 3 - remainder
                operations += min(remainder, 3 - remainder)
        
        # Return the total count of operations
        return operations