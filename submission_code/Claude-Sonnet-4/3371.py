class Solution:
    def sumOfTheDigitsOfHarshadNumber(self, x: int) -> int:
        # Calculate sum of digits
        digit_sum = 0
        temp = x
        while temp > 0:
            digit_sum += temp % 10
            temp //= 10
        
        # Check if x is divisible by sum of its digits
        if x % digit_sum == 0:
            return digit_sum
        else:
            return -1