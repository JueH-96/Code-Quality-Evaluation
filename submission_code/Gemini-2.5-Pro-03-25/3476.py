import math
from typing import List

class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        """
        Calculates the minimum number of operations (add/subtract 1)
        to make all elements in nums divisible by 3.

        Args:
            nums: A list of integers.

        Returns:
            The minimum total number of operations required.
        """
        operations_count = 0
        
        # Iterate through each number in the list
        for num in nums:
            # Calculate the remainder when the number is divided by 3
            remainder = num % 3
            
            # If the remainder is 0, the number is already divisible by 3,
            # so 0 operations are needed for this number.
            if remainder == 0:
                continue 
            # If the remainder is 1, we need 1 operation (subtract 1) to make it divisible by 3 (e.g., 4 -> 3).
            # Adding 2 (e.g., 4 -> 6) would take 2 operations, which is not minimal.
            elif remainder == 1:
                operations_count += 1
            # If the remainder is 2, we need 1 operation (add 1) to make it divisible by 3 (e.g., 5 -> 6).
            # Subtracting 2 (e.g., 5 -> 3) would take 2 operations, which is not minimal.
            elif remainder == 2:
                operations_count += 1
                
            # In summary: if the number is not divisible by 3 (remainder is 1 or 2), 
            # we always need exactly one operation (either add 1 or subtract 1) 
            # to reach the nearest multiple of 3.
            
        return operations_count

    # # Alternative concise solution using sum and generator expression:
    # def minimumOperations(self, nums: List[int]) -> int:
    #     """
    #     Calculates the minimum operations using a more concise approach.
    #     """
    #     # Count how many numbers are not divisible by 3.
    #     # Each such number requires exactly one operation.
    #     return sum(1 for num in nums if num % 3 != 0)