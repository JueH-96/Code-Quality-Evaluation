class Solution:
    def sumOfTheDigitsOfHarshadNumber(self, x: int) -> int:
        # Convert the integer to a string to access its digits
        digits = [int(d) for d in str(x)]
        
        # Calculate the sum of the digits
        digit_sum = sum(digits)
        
        # Check if the number is a Harshad number
        if x % digit_sum == 0:
            return digit_sum
        else:
            return -1