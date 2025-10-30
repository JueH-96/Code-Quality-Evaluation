class Solution:
    def sumOfTheDigitsOfHarshadNumber(self, x: int) -> int:
        # Calculate the sum of the digits of x
        num = x
        sum_of_digits = 0
        while num > 0:
            sum_of_digits += num % 10
            num = num // 10
        
        # Check if x is divisible by the sum of its digits
        if x % sum_of_digits == 0:
            return sum_of_digits
        else:
            return -1