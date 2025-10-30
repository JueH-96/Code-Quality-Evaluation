class Solution:
    def sumOfTheDigitsOfHarshadNumber(self, x: int) -> int:
        # Calculate the sum of the digits of x
        sum_digits = sum(int(digit) for digit in str(x))
        # Check if x is divisible by the sum of its digits
        if x % sum_digits == 0:
            return sum_digits
        else:
            return -1