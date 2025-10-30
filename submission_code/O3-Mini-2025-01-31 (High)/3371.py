class Solution:
    def sumOfTheDigitsOfHarshadNumber(self, x: int) -> int:
        # Calculate the sum of the digits of x
        digits_sum = sum(int(d) for d in str(x))
        
        # Check if x is divisible by the sum of its digits (Harshad Number condition)
        if x % digits_sum == 0:
            return digits_sum
        else:
            return -1