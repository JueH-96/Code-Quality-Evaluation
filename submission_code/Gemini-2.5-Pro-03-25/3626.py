import math

class Solution:
    """
    Finds the smallest number >= n whose digit product is divisible by t.
    """

    def _calculate_digit_product(self, num: int) -> int:
        """
        Helper function to calculate the product of digits of a non-negative integer.
        Returns 0 if any digit is 0.
        """
        # Base case for single digit numbers or the number 0 itself.
        if num == 0:
            return 0
            
        # Convert the number to a string to easily iterate through digits.
        s_num = str(num)

        # Optimization: If '0' is present as a digit, the product is 0.
        # Check this first as it's a common case and simplifies calculation.
        if '0' in s_num:
            return 0

        # Calculate the product of non-zero digits.
        product = 1
        for digit_char in s_num:
            # No need to handle '0' here due to the check above.
            digit = int(digit_char)
            product *= digit
            
        return product

    def smallestNumber(self, n: int, t: int) -> int:
        """
        Finds the smallest number >= n such that the product of its digits 
        is divisible by t.

        Args:
            n: The lower bound for the number search (inclusive). 1 <= n <= 100.
            t: The divisor for the digit product. 1 <= t <= 10.

        Returns:
            The smallest integer x >= n where the product of digits of x 
            is divisible by t.
        """
        
        # Start checking from n upwards.
        current_num = n
        
        while True:
            # Calculate the product of digits for the current number.
            product = self._calculate_digit_product(current_num)

            # Check if the product is divisible by t.
            # The modulo operator (%) handles divisibility check.
            # If product is 0 (because the number contains a '0' digit), 
            # 0 % t will be 0 for any t >= 1, satisfying the condition.
            if product % t == 0:
                # We found the smallest number satisfying the condition
                # because we started from n and incremented.
                return current_num

            # If the condition is not met, increment the number and check again.
            current_num += 1

            # This loop is guaranteed to terminate. Eventually, we will encounter 
            # a number containing the digit '0' (e.g., 10, 20, ..., 100, 110...). 
            # The product of digits for such a number is 0. 
            # Since 0 is divisible by any integer t >= 1, the condition 
            # product % t == 0 will eventually be met.