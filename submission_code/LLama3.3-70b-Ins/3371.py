class Solution:
    def sumOfTheDigitsOfHarshadNumber(self, x: int) -> int:
        # Calculate the sum of the digits of x
        sum_of_digits = sum(int(digit) for digit in str(x))
        
        # Check if x is divisible by the sum of its digits
        if x % sum_of_digits == 0:
            # If x is a Harshad number, return the sum of its digits
            return sum_of_digits
        else:
            # If x is not a Harshad number, return -1
            return -1