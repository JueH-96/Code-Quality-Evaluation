class Solution:
    def sumOfTheDigitsOfHarshadNumber(self, x: int) -> int:
        # Compute the sum of the digits of x
        digit_sum = sum(int(d) for d in str(x))
        # Check if x is divisible by the sum of its digits
        if x % digit_sum == 0:
            return digit_sum
        else:
            return -1