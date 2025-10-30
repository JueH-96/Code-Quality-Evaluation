class Solution:
    def sumOfTheDigitsOfHarshadNumber(self, x: int) -> int:
        # Compute the sum of digits
        digit_sum = sum(int(d) for d in str(x))
        # Check for Harshad number and return accordingly
        if x % digit_sum == 0:
            return digit_sum
        else:
            return -1