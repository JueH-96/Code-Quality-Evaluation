class Solution:
    """
    Calculates the difference between the sum of numbers not divisible by m
    and the sum of numbers divisible by m in the range [1, n].
    """
    def differenceOfSums(self, n: int, m: int) -> int:
        """
        Finds the difference between the sum of integers in [1, n] not divisible by m (num1)
        and the sum of integers in [1, n] divisible by m (num2).

        Args:
            n: The upper limit of the range (inclusive). Positive integer.
            m: The divisor. Positive integer.

        Returns:
            The calculated difference num1 - num2.
        """
        num1 = 0  # Initialize the sum of integers not divisible by m
        num2 = 0  # Initialize the sum of integers divisible by m

        # Iterate through all integers from 1 to n (inclusive)
        for i in range(1, n + 1):
            # Check if the current integer i is divisible by m
            if i % m == 0:
                # If i is divisible by m, add it to num2
                num2 += i
            else:
                # If i is not divisible by m, add it to num1
                num1 += i

        # Calculate the final difference
        difference = num1 - num2
        
        # Return the result
        return difference