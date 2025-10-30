class Solution:
    def sumOfTheDigitsOfHarshadNumber(self, x: int) -> int:
        # Convert the integer to a string to easily access each digit
        str_x = str(x)
        
        # Calculate the sum of the digits
        sum_of_digits = sum(int(digit) for digit in str_x)
        
        # Check if the number is divisible by the sum of its digits
        if x % sum_of_digits == 0:
            return sum_of_digits
        else:
            return -1