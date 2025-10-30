from typing import List

class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        """
        Calculates the minimum number of operations to make all elements
        of nums divisible by 3.

        An operation consists of adding or subtracting 1 from any element.

        The minimum operations for a single number 'x' to be divisible by 3
        is determined by its remainder when divided by 3:
        - If x % 3 == 0: 0 operations needed.
        - If x % 3 == 1: 1 operation needed (subtract 1).
        - If x % 3 == 2: 1 operation needed (add 1).

        This is because the closest multiple of 3 is always within 1 step
        if the remainder is 1 or 2.

        The total minimum operations is the sum of minimum operations
        needed for each element.

        Args:
            nums: A list of integers.

        Returns:
            The minimum total number of operations.
        """
        total_operations = 0
        for num in nums:
            # If the number is not divisible by 3, it requires 1 operation.
            # If it is divisible by 3, it requires 0 operations.
            if num % 3 != 0:
                total_operations += 1
                
        return total_operations