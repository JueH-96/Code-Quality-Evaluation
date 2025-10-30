class Solution:
    def sumOfTheDigitsOfHarshadNumber(self, x: int) -> int:
        # Calculate the sum of the digits of x.
        # For example, if x = 18, str(x) is "18".
        # The generator (int(digit) for digit in "18") yields 1, then 8.
        # sum() will compute 1 + 8 = 9.
        sum_of_digits = sum(int(digit) for digit in str(x))
        
        # Check if x is divisible by the sum of its digits.
        # According to the problem constraints (1 <= x <= 100), x is positive,
        # so sum_of_digits will also be positive (at least 1).
        # Thus, there's no risk of ZeroDivisionError.
        if x % sum_of_digits == 0:
            # If x is divisible by sum_of_digits, it's a Harshad number.
            # Return the sum_of_digits.
            return sum_of_digits
        else:
            # Otherwise, it's not a Harshad number.
            # Return -1.
            return -1