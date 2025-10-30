class Solution:
    def sumOfTheDigitsOfHarshadNumber(self, x: int) -> int:
        sum_digits = sum(int(digit) for digit in str(x))
        if x % sum_digits == 0:
            return sum_digits
        else:
            return -1