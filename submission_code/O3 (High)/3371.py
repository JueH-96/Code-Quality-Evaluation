class Solution:
    def sumOfTheDigitsOfHarshadNumber(self, x: int) -> int:
        # Calculate sum of digits
        digit_sum = sum(int(ch) for ch in str(x))
        
        # Check Harshad condition
        if x % digit_sum == 0:
            return digit_sum
        else:
            return -1