class Solution:
    def sumOfTheDigitsOfHarshadNumber(self, x: int) -> int:
        # Calculate the sum of digits of x
        digits_sum = sum(int(d) for d in str(x))
        
        # Check if x is divisible by its digits_sum
        if x % digits_sum == 0:
            return digits_sum
        else:
            return -1