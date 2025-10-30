class Solution:
    def sumOfTheDigitsOfHarshadNumber(self, x: int) -> int:
        # Calculate the sum of digits
        # Convert the integer to a string to iterate through digits
        # The sum() function combined with a generator expression is a concise way to do this.
        digit_sum = sum(int(digit) for digit in str(x))

        # Check if x is divisible by the sum of its digits.
        # A number is divisible by another if the remainder of their division is 0.
        # Since the input constraint is 1 <= x <= 100, x is always positive.
        # The sum of digits for a positive number is always at least 1 (e.g., sum of digits of 1 is 1).
        # Thus, digit_sum will always be >= 1, preventing division by zero errors.
        if x % digit_sum == 0:
            # If x is divisible by the sum of its digits, it is a Harshad number.
            # Return the sum of the digits.
            return digit_sum
        else:
            # If x is not divisible by the sum of its digits, it is not a Harshad number.
            # Return -1 as per the problem specification.
            return -1