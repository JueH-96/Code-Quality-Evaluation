from typing import List

class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        # Initialize a counter for the total minimum operations.
        total_operations = 0
        
        # Iterate through each number in the input array.
        for num in nums:
            # Calculate the remainder when the number is divided by 3.
            remainder = num % 3
            
            # If the remainder is 0, the number is already divisible by 3, 
            # so no operations are needed for this number.
            if remainder == 0:
                continue
            # If the remainder is 1 or 2, the number is not divisible by 3.
            # In either case (remainder 1 or 2), exactly one operation is sufficient 
            # and minimal to make the number divisible by 3.
            # For example:
            #   If remainder is 1 (e.g., num=4): Subtract 1 (4 -> 3). Cost = 1.
            #   If remainder is 2 (e.g., num=2): Add 1 (2 -> 3). Cost = 1.
            else: # remainder is 1 or 2
                total_operations += 1
                
        # Return the accumulated total minimum operations.
        return total_operations