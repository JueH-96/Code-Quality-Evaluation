class Solution:
    def sumOfTheDigitsOfHarshadNumber(self, x: int) -> int:
        # Step 1: Calculate the sum of the digits of x.
        # We'll use a temporary variable `temp_x` to extract digits without modifying the original `x`.
        sum_digits = 0
        temp_x = x
        
        # Loop while temp_x is greater than 0 to process each digit.
        while temp_x > 0:
            # Get the last digit by using the modulo operator (%).
            digit = temp_x % 10
            # Add the digit to our sum.
            sum_digits += digit
            # Remove the last digit by using integer division (//).
            temp_x //= 10
        
        # Step 2: Check if x is a Harshad number.
        # A number is Harshad if it is divisible by the sum of its digits.
        if x % sum_digits == 0:
            # If x is a Harshad number, return the calculated sum of its digits.
            return sum_digits
        else:
            # If x is not a Harshad number, return -1 as per the problem specification.
            return -1