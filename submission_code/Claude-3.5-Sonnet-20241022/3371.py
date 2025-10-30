class Solution:
    def sumOfTheDigitsOfHarshadNumber(self, x: int) -> int:
        # Calculate sum of digits
        digit_sum = sum(int(d) for d in str(x))
        
        # Check if x is divisible by sum of its digits
        if x % digit_sum == 0:
            return digit_sum
        return -1